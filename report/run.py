from pathlib import Path

from report.gradio_app import create_gradio_app
from text_dedup.config.base import Config


def main(config: Config) -> None:
    """Launch Gradio app with pre-configured settings"""
    app = create_gradio_app()
    output_dir = Path(config.output.output_dir)
    if output_dir.exists():
        for component in app.blocks.values():
            if not hasattr(component, "label") or not hasattr(component, "value"):
                continue
            if component.label == "Output Directory Path":
                component.value = str(output_dir)
            elif component.label == "Text Column Name":
                component.value = config.algorithm.text_column
            elif component.label == "Cluster Column Name":
                component.value = config.algorithm.cluster_column
            elif component.label == "Internal Index Column Name":
                component.value = config.algorithm.internal_index_column

    app.launch(share=False)


if __name__ == "__main__":
    from pydantic_settings import CliApp

    config = CliApp.run(Config)
    main(config)
