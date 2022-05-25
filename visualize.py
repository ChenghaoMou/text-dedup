from __future__ import annotations

import pandas as pd
import streamlit as st


def load_data(file: str = 'results.jsonl'):

    df = pd.read_json(file, lines=True)

    return df


df = load_data('minhash_results.jsonl')

if 'substring' in df.columns:

    count = 0
    prev_id = None
    for row in df.to_records('dict'):
        if row['query_id'] != prev_id:
            st.markdown('---')
            prev_id = row['query_id']

        result = row['query']

        start = result.index(row['substring'])
        st.markdown(f'...{result[start-100:start]}<mark style="background-color:Tomato;">{result[start:start+len(row["substring"])]}</mark>{result[start+len(row["substring"]):start+len(row["substring"])+100]}...', unsafe_allow_html=True)

        count += 1
        if count > 100:
            break

if 'ref' in df.columns:
    count = 0
    prev_id = None
    for row in df.to_records('dict'):
        if row['query_id'] != prev_id:
            st.markdown('---')
            prev_id = row['query_id']

        x = row['query']
        y = row['ref']

        st.markdown(f'`{row["query_id"]}`')
        st.markdown(f'{x}', unsafe_allow_html=True)

        st.markdown(f'`{row["ref_id"]}`')
        st.markdown(f'{y}', unsafe_allow_html=True)

        count += 1
        if count > 100:
            break
