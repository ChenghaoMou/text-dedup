import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datasets import Dataset
from datasets import load_from_disk
from gradio_rangeslider import RangeSlider
from plotly.subplots import make_subplots

from text_dedup.config import MinHashAlgorithmConfig
from text_dedup.config.base import Config
from text_dedup.utils.jaccard import cluster_jaccard_similarity


class ClusterVisualizer:
    def __init__(self) -> None:
        self.dataset: Dataset | None = None
        self.cluster2ids: dict | None = None
        self.config: Config | None = None
        self.max_cluster_size: int = 100  # Default, will be updated when dataset loads
        self.cluster_sizes: list = []

    def load_dataset(
        self,
        output_dir: str,
        text_column: str = "text",
        cluster_column: str = "__CLUSTER__",
        index_column: str = "__INDEX__",
    ) -> tuple[pd.DataFrame | None, str, dict[str, Any]]:
        """Load dataset and compute cluster statistics"""
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return None, "Directory does not exist", gr.update()

            if not (output_path.parent / "config.toml").exists():
                return None, "Config file not found", gr.update()

            self.config = Config()  # type: ignore[call-arg]

            ds = load_from_disk(output_path)
            if not isinstance(ds, Dataset):
                return None, "Not a valid dataset", gr.update()

            self.dataset = ds
            self.text_column = text_column
            self.cluster_column = cluster_column
            with open(output_path / "clusters.pickle", "rb") as f:
                PARENTS: dict[int, int] = pickle.load(f)  # noqa: S301

            if cluster_column not in self.dataset.column_names:
                return None, f"Cluster column '{cluster_column}' not found in dataset", gr.update()
            if text_column not in self.dataset.column_names:
                return None, f"Text column '{text_column}' not found in dataset", gr.update()

            id2cluster = dict(
                zip(range(len(self.dataset)), [PARENTS.get(i, i) for i in self.dataset[index_column]], strict=False)
            )
            self.cluster2ids = defaultdict(list)
            for _id, cluster in id2cluster.items():
                self.cluster2ids[cluster].append(_id)

            self.cluster_sizes = [len(ids) for ids in self.cluster2ids.values()]
            self.max_cluster_size = max(self.cluster_sizes, default=100)

            stats = self.get_summary_stats()
            slider_update = gr.update(maximum=self.max_cluster_size, value=(1, min(100, self.max_cluster_size)))

        except Exception as e:
            print(e)
            return None, f"Error loading dataset: {e!s}", gr.update()
        else:
            return stats, "Dataset loaded successfully!", slider_update

    def get_summary_stats(self) -> pd.DataFrame | None:
        """Get summary statistics about the dataset"""
        if not self.dataset or not self.cluster2ids:
            return None

        total_records = len(self.dataset)
        total_clusters = len(self.cluster2ids)
        unique_records = self.cluster_sizes.count(1)
        duplicate_records = total_records - unique_records
        duplicate_clusters = total_clusters - unique_records

        stats = {
            "Total Records": f"{total_records:,}",
            "Total Clusters": f"{total_clusters:,}",
            "Unique Records (cluster size = 1)": f"{unique_records:,}",
            "Number of Duplicate Clusters": f"{duplicate_clusters:,}",
            "Records in Duplicate Clusters": f"{duplicate_records:,}",
            "Deduplication Rate": f"{(1 - total_clusters / total_records) * 100:.2f}%",
            "Average Cluster Size": f"{sum(self.cluster_sizes) / len(self.cluster_sizes):.2f}",
            "Max Cluster Size": f"{self.max_cluster_size:,}",
            "Min Cluster Size": f"{min(self.cluster_sizes):,}",
        }

        return pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

    def plot_cluster_distribution(self, cluster_size_slider: tuple[int, int]) -> go.Figure | None:
        """Plot interactive histogram of cluster sizes using Plotly"""
        if not self.cluster2ids:
            return None

        min_size, max_size = cluster_size_slider

        df = pd.DataFrame({"cluster_id": list(self.cluster2ids.keys()), "size": self.cluster_sizes})

        df_filtered = df[(df["size"] >= min_size) & (df["size"] <= max_size)]
        total_clusters_in_range = len(df_filtered)
        total_records_in_range = df_filtered["size"].sum()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Cluster Size Distribution ({int(min_size)} - {int(max_size)})<br>"
                f"<sup>{total_clusters_in_range:,} clusters containing {total_records_in_range:,} records</sup>",
                "Cluster Size Distribution (Log-Log Scale)<br>"
                "<sup>All clusters shown, filtered range highlighted</sup>",
            ),
            horizontal_spacing=0.15,
        )

        if len(df_filtered) > 0:
            histogram = go.Histogram(
                x=df_filtered["size"],
                nbinsx=int(min(50, max_size - min_size + 1)),
                name="Cluster Size Distribution",
                marker_color="rgba(0, 123, 255, 0.7)",
                marker_line_color="rgba(0, 0, 0, 0.8)",
                marker_line_width=1,
                hovertemplate="Size: %{x}<br>Count: %{y}<extra></extra>",
            )
            fig.add_trace(histogram, row=1, col=1)
        else:
            fig.add_annotation(
                text="No clusters in this size range",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=1,
                col=1,
            )

        size_counts = df["size"].value_counts().sort_index()

        scatter_all = go.Scatter(
            x=size_counts.index,
            y=size_counts.values,
            mode="markers",
            marker={"size": 4, "color": "rgba(200, 200, 200, 0.5)"},
            line={"color": "rgba(200, 200, 200, 0.3)", "width": 1},
            name="All Clusters",
            hovertemplate="Size: %{x}<br>Frequency: %{y}<extra></extra>",
        )
        fig.add_trace(scatter_all, row=1, col=2)

        size_counts_filtered = df_filtered["size"].value_counts().sort_index()
        if len(size_counts_filtered) > 0:
            scatter_filtered = go.Scatter(
                x=size_counts_filtered.index,
                y=size_counts_filtered.values,
                mode="markers",
                marker={"size": 8, "color": "rgba(255, 99, 71, 0.8)"},
                line={"color": "rgba(255, 99, 71, 0.5)", "width": 2},
                name="Filtered Range",
                hovertemplate="Size: %{x}<br>Frequency: %{y}<extra></extra>",
            )
            fig.add_trace(scatter_filtered, row=1, col=2)

        fig.update_xaxes(title_text="Cluster Size", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        fig.update_xaxes(title_text="Cluster Size (log scale)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (log scale)", type="log", row=1, col=2)

        fig.add_vrect(
            x0=min_size,
            x1=max_size,
            fillcolor="rgba(0, 123, 255, 0.1)",
            layer="below",
            line_width=0,
            annotation_text=f"Filter: {min_size}-{max_size}",
            annotation_position="top left",
            row=1,
            col=2,
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            legend={"x": 0.7, "y": 0.95},
            title_text="Cluster Size Distribution Analysis",
            title_x=0.5,
            template="plotly_white",
        )

        return fig

    def plot_detailed_distribution(self, bin_size: int = 10) -> go.Figure | None:
        """Create additional interactive visualizations"""
        if not self.cluster2ids:
            return None

        df = pd.DataFrame({"cluster_id": list(self.cluster2ids.keys()), "size": self.cluster_sizes})

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cluster Size Box Plot",
                "Cumulative Distribution",
                "Size Range Distribution",
                "Top 50 Clusters",
            ),
            specs=[[{"type": "box"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        box = go.Box(
            y=df["size"],
            name="Cluster Sizes",
            boxpoints="outliers",
            marker_color="rgba(0, 123, 255, 0.7)",
            hovertemplate="Size: %{y}<extra></extra>",
        )
        fig.add_trace(box, row=1, col=1)

        df_sorted = df.sort_values("size")
        cumulative = go.Scatter(
            x=df_sorted["size"],
            y=list(range(1, len(df_sorted) + 1)),
            mode="lines",
            name="Cumulative",
            line={"color": "rgba(255, 99, 71, 0.8)", "width": 2},
            hovertemplate="Size: %{x}<br>Cumulative Count: %{y}<extra></extra>",
        )
        fig.add_trace(cumulative, row=1, col=2)

        bins = pd.cut(df["size"], bins=range(1, max(df["size"]) + bin_size, bin_size))
        bin_counts = bins.value_counts().sort_index()
        bar1 = go.Bar(
            y=[str(interval) for interval in bin_counts.index],
            x=bin_counts.values,
            name="Size Ranges",
            marker_color="rgba(46, 204, 113, 0.7)",
            hovertemplate="Range: %{y}<br>Count: %{x}<extra></extra>",
            orientation="h",
        )
        fig.add_trace(bar1, row=2, col=1)

        top_50 = df.nlargest(min(50, len(df)), "size")
        bar2 = go.Bar(
            x=[f"Cluster {i}" for i in range(1, len(top_50) + 1)],
            y=top_50["size"].values,
            name="Top Clusters",
            marker_color="rgba(231, 76, 60, 0.7)",
            hovertemplate="Rank: %{x}<br>Size: %{y}<br>Cluster ID: %{customdata}<extra></extra>",
            customdata=top_50["cluster_id"].values,
        )
        fig.add_trace(bar2, row=2, col=2)

        fig.update_yaxes(title_text="Cluster Size", row=1, col=1)
        fig.update_xaxes(title_text="Cluster Size", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Count", row=1, col=2)
        fig.update_xaxes(title_text="Size Range", tickangle=-45, row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Cluster Rank", tickangle=-45, row=2, col=2)
        fig.update_yaxes(title_text="Size", row=2, col=2)

        fig.update_layout(
            height=800, showlegend=False, title_text="Detailed Cluster Analysis", title_x=0.5, template="plotly_white"
        )

        return fig

    def plot_cluster_treemap(self, max_clusters: int = 100) -> go.Figure | None:
        """Create a treemap visualization of cluster sizes"""
        if not self.cluster2ids:
            return None
        if not isinstance(self.dataset, Dataset):
            return None

        top_clusters = sorted(self.cluster2ids.items(), key=lambda x: len(x[1]), reverse=True)[:max_clusters]

        labels = ["All Clusters"]
        parents = [""]
        values = [0]
        text = [f"Total: {sum(len(ids) for _, ids in top_clusters):,} records"]

        for cluster_id, ids in top_clusters:
            sample_text = self.dataset[ids[0]][self.text_column][:50] + "..."
            labels.append(f"Cluster {cluster_id}")
            parents.append("All Clusters")
            values.append(len(ids))
            text.append(f"Size: {len(ids):,}<br>Sample: {sample_text}")

        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                text=text,
                textinfo="label+value",
                hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
                marker={"colorscale": "Blues", "reversescale": False, "line": {"width": 1}},
            )
        )

        fig.update_layout(title=f"Treemap of Top {max_clusters} Clusters by Size", height=600, template="plotly_white")

        return fig

    def get_top_clusters(self, n: int = 20) -> pd.DataFrame | None:
        """Get top N clusters by size"""
        if not self.cluster2ids:
            return None
        if not isinstance(self.dataset, Dataset):
            return None

        top_clusters = sorted(self.cluster2ids.items(), key=lambda x: len(x[1]), reverse=True)[:n]

        data = []
        for rank, (cluster_id, ids) in enumerate(top_clusters, 1):
            data.append({
                "Rank": rank,
                "Cluster ID": cluster_id,
                "Size": f"{len(ids):,}",
                "Sample Text": self.dataset[ids[0]][self.text_column][:100] + "...",
            })

        return pd.DataFrame(data)

    def explore_cluster(self, cluster_id: int, max_samples: int = 10) -> tuple[pd.DataFrame | None, str]:
        """Get samples from a specific cluster"""
        if not self.cluster2ids or not self.dataset:
            return None, "No dataset loaded"

        if cluster_id not in self.cluster2ids:
            return None, f"Cluster {cluster_id} not found"

        ids = self.cluster2ids[cluster_id]
        cluster_size = len(ids)

        sample_ids = ids[:max_samples]
        samples = []

        for i, idx in enumerate(sample_ids):
            text = self.dataset[idx][self.text_column]
            if len(text) > 500:
                text = text[:500] + "..."
            samples.append({"Sample #": i + 1, "Record ID": idx, "Text Preview": text})

        df = pd.DataFrame(samples)
        info = f"Cluster {cluster_id}: {cluster_size:,} records total (showing {len(samples)} samples)"

        if self.config and isinstance(self.config.algorithm, MinHashAlgorithmConfig):
            tokenize = self.config.algorithm.get_ngrams_func()
            docs = self.dataset.select(sample_ids)[self.text_column]
            similarities, fp_rate = cluster_jaccard_similarity(
                cluster=[tokenize(doc.lower()) for doc in docs], threshold=self.config.algorithm.threshold
            )
            info += "\n\n##Sample jaccard similarity\n\n"
            info += f"- Max: {max(similarities):.4f}\n"
            info += f"- Mean: {np.mean(similarities):.4f}\n"
            info += f"- Min: {min(similarities):.4f}\n"
            info += f"- 90 Percentile: {np.percentile(similarities, 0.9):.4f}\n"
            info += f"- FP Rate: {fp_rate:.4f}"

        return df, info

    def search_text(self, query: str, max_results: int = 20) -> tuple[pd.DataFrame | None, str]:
        """Search for text across all records"""
        if not self.dataset:
            return None, "No dataset loaded"
        if not isinstance(self.dataset, Dataset):
            return None, "No dataset loaded"
        if not self.cluster2ids:
            return None, "No clusters loaded"

        query_lower = query.lower()
        results = []

        for idx, text in enumerate(self.dataset[self.text_column]):
            if query_lower in text.lower():
                cluster_id = self.dataset[idx][self.cluster_column]
                results.append({
                    "Record ID": idx,
                    "Cluster ID": cluster_id,
                    "Cluster Size": f"{len(self.cluster2ids[cluster_id]):,}",
                    "Text Preview": text[:200] + "...",
                })
                if len(results) >= max_results:
                    break

        if not results:
            return None, "No results found"

        return pd.DataFrame(results), f"Found {len(results)} results"

    def compare_clusters(
        self, cluster_id_1: int, cluster_id_2: int, max_samples: int = 10
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
        """Compare two clusters by showing samples from each."""
        if not self.cluster2ids or not self.dataset:
            return None, None, "No dataset loaded"

        info: list[str] = []
        dfs: list[pd.DataFrame | None] = []
        for cid in [cluster_id_1, cluster_id_2]:
            if cid not in self.cluster2ids:
                dfs.append(None)
                info.append(f"Cluster {cid} not found")
            else:
                ids = self.cluster2ids[cid]
                samples = []
                for i, idx in enumerate(ids[:max_samples]):
                    text = self.dataset[idx][self.text_column]
                    if len(text) > 500:
                        text = text[:500] + "..."
                    samples.append({"Sample #": i + 1, "Record ID": idx, "Text Preview": text})
                dfs.append(pd.DataFrame(samples))
                info.append(f"Cluster {cid}: {len(ids):,} records (showing {min(len(ids), max_samples)})")
        return dfs[0], dfs[1], " | ".join(info)


def create_gradio_app() -> gr.Blocks:
    visualizer = ClusterVisualizer()

    with gr.Blocks(
        title="Text Deduplication Cluster Visualizer",
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans"), "IBM Plex Sans", "sans-serif"]),
    ) as app:
        gr.Markdown("# Text Deduplication Cluster Visualizer")
        gr.Markdown("Explore and visualize text deduplication results with interactive plots.")

        with gr.Tab("Dataset Loading"):
            with gr.Row():
                with gr.Column():
                    output_dir = gr.Textbox(label="Output Directory Path", placeholder="e.g., ./output", value="output")
                    text_column = gr.Textbox(label="Text Column Name", value="text")
                    cluster_column = gr.Textbox(label="Cluster Column Name", value="__CLUSTER__")
                    index_column = gr.Textbox(label="Internal Index Column Name", value="__INDEX__")
                    load_btn = gr.Button("Load Dataset", variant="primary")

                with gr.Column():
                    load_status = gr.Textbox(label="Status", interactive=False)
                    summary_stats = gr.Dataframe(label="Dataset Summary Statistics")

            with gr.Tab("Cluster Distribution"):
                with gr.Row():
                    cluster_size_slider = RangeSlider(
                        minimum=1, maximum=1000, value=(2, 100), step=1, label="Cluster Size Range"
                    )

                distribution_plot = gr.Plot()

                cluster_size_slider.change(
                    fn=visualizer.plot_cluster_distribution, inputs=[cluster_size_slider], outputs=[distribution_plot]
                )

            load_btn.click(
                fn=lambda directory, text_col, cluster_col, index_col: visualizer.load_dataset(
                    directory, text_col, cluster_col, index_col
                ),
                inputs=[output_dir, text_column, cluster_column, index_column],
                outputs=[summary_stats, load_status, cluster_size_slider],
            )

        with gr.Tab("Detailed Analysis"):
            with gr.Row():
                with gr.Column(scale=3):
                    bin_size_slider = gr.Slider(
                        minimum=5, maximum=50, value=10, step=5, label="Bin Size for Range Distribution"
                    )
                with gr.Column(scale=1):
                    analyze_btn = gr.Button("Generate Analysis", variant="primary")

            detailed_plot = gr.Plot(label="Detailed Cluster Analysis")

            analyze_btn.click(
                fn=visualizer.plot_detailed_distribution, inputs=[bin_size_slider], outputs=[detailed_plot]
            )

        with gr.Tab("Treemap Visualization"):
            with gr.Row():
                with gr.Column(scale=3):
                    max_treemap_clusters = gr.Slider(
                        minimum=20, maximum=200, value=100, step=10, label="Number of Clusters to Show"
                    )
                with gr.Column(scale=1):
                    treemap_btn = gr.Button("Generate Treemap", variant="primary")

            treemap_plot = gr.Plot()

            treemap_btn.click(fn=visualizer.plot_cluster_treemap, inputs=[max_treemap_clusters], outputs=[treemap_plot])

        with gr.Tab("Top Clusters"):
            with gr.Row():
                with gr.Column(scale=3):
                    top_n = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Number of Top Clusters to Show")
                with gr.Column(scale=1):
                    refresh_top_btn = gr.Button("Refresh Top Clusters", variant="primary")

            top_clusters_df = gr.Dataframe(label="Top Clusters by Size", wrap=True)

            refresh_top_btn.click(fn=visualizer.get_top_clusters, inputs=[top_n], outputs=[top_clusters_df])

        with gr.Tab("Cluster Explorer"):
            with gr.Row():
                with gr.Column():
                    cluster_id_input = gr.Number(label="Cluster ID", precision=0)
                    max_samples_slider = gr.Slider(
                        minimum=1, maximum=500, value=10, step=1, label="Maximum Samples to Show"
                    )
                    explore_btn = gr.Button("Explore Cluster", variant="primary")

                with gr.Column():
                    cluster_info = gr.Markdown(label="Cluster Information")

            cluster_samples = gr.Dataframe(label="Cluster Samples", wrap=True)

            explore_btn.click(
                fn=visualizer.explore_cluster,
                inputs=[cluster_id_input, max_samples_slider],
                outputs=[cluster_samples, cluster_info],
            )

        with gr.Tab("Text Search"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter text to search...")
                    max_search_results = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Maximum Results")
                    search_btn = gr.Button("Search", variant="primary")

                with gr.Column():
                    search_status = gr.Textbox(label="Search Status", interactive=False)

            search_results = gr.Dataframe(label="Search Results", wrap=True)

            search_btn.click(
                fn=visualizer.search_text,
                inputs=[search_query, max_search_results],
                outputs=[search_results, search_status],
            )

        with gr.Tab("Cluster Comparison"):
            with gr.Row():
                with gr.Column():
                    cluster_id_1 = gr.Number(label="Cluster ID 1", precision=0)
                    cluster_id_2 = gr.Number(label="Cluster ID 2", precision=0)
                    max_samples_compare = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1, label="Maximum Samples to Show"
                    )
                    compare_btn = gr.Button("Compare Clusters", variant="primary")

                with gr.Column():
                    compare_status = gr.Textbox(label="Comparison Info", interactive=False)

            with gr.Row():
                cluster1_samples = gr.Dataframe(label="Cluster 1 Samples", wrap=True)
                cluster2_samples = gr.Dataframe(label="Cluster 2 Samples", wrap=True)

            compare_btn.click(
                fn=visualizer.compare_clusters,
                inputs=[cluster_id_1, cluster_id_2, max_samples_compare],
                outputs=[cluster1_samples, cluster2_samples, compare_status],
            )

        return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
