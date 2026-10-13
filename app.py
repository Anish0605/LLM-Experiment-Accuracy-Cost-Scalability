                a_delta = float(by_method["LLM-assisted"]["field_accuracy_pct"] - by_method["Rule-based"]["field_accuracy_pct"])
                b.metric("Rule field accuracy", f"{by_method['Rule-based']['field_accuracy_pct']:.2f}%")
                c.metric("LLM field accuracy", f"{by_method['LLM-assisted']['field_accuracy_pct']:.2f}%", delta=f"{a_delta:+.2f} pp")
                llm_cost = by_method["LLM-assisted"]["estimated_service_cost_usd"]
                d.metric("LLM service-equivalent cost", f"${llm_cost:.6f}" if pd.notna(llm_cost) else "Unavailable")
                st.caption("Cost is token-count × reference unit price, not a billing statement. A provider free allowance can make actual expenditure $0.")

                chart1, chart2 = st.columns(2)
                with chart1:
                    acc_long = summary_df.melt(
                        id_vars=["sample_size", "method"],
                        value_vars=["field_accuracy_pct", "record_accuracy_pct"],
                        var_name="measure", value_name="accuracy_pct",
                    )
                    acc_long["measure"] = acc_long["measure"].map({"field_accuracy_pct": "Field accuracy", "record_accuracy_pct": "Complete-record accuracy"})
                    fig = px.bar(acc_long, x="measure", y="accuracy_pct", color="method", barmode="group", facet_col="sample_size", title="Accuracy on completed sample sizes", range_y=[0, 100], labels={"measure": "Measure", "accuracy_pct": "Accuracy (%)", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 400), use_container_width=True)
                with chart2:
                    fig = px.line(summary_df, x="sample_size", y="processing_seconds", color="method", markers=True, title="Observed processing time", labels={"sample_size": "Records tested", "processing_seconds": "Observed seconds", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 400), use_container_width=True)

                chart3, chart4 = st.columns(2)
                with chart3:
                    cost_df = summary_df[summary_df["method"] == "LLM-assisted"].copy()
                    if not cost_df.empty and cost_df["estimated_service_cost_usd"].notna().any():
                        fig = px.bar(cost_df, x="sample_size", y="estimated_service_cost_usd", title="LLM service-equivalent cost by tested sample size", labels={"sample_size": "Records actually tested", "estimated_service_cost_usd": "Estimated cost (USD)"})
                        st.plotly_chart(style_plot(fig, 330), use_container_width=True)
                    else:
                        st.info("Token usage was not returned for these calls, so cost estimates are unavailable.")
                with chart4:
                    throughput_df = summary_df.copy()
                    fig = px.line(throughput_df, x="sample_size", y="records_per_second", color="method", markers=True, title="Observed throughput", labels={"sample_size": "Records actually tested", "records_per_second": "Records / second", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 330), use_container_width=True)

                st.markdown("#### Summary table")
                display_cols = ["sample_size", "method", "field_accuracy_pct", "record_accuracy_pct", "processing_seconds", "records_per_second", "input_tokens", "output_tokens", "estimated_service_cost_usd", "attempted_api_calls", "api_errors"]
                st.dataframe(summary_df[display_cols].round(6), use_container_width=True, hide_index=True)
                st.caption("Times are measured, not extrapolated. Rule time measures local Python transformation time; LLM time sums API request wall time, including provider/network response. This is not a fully matched hardware/runtime comparison.")

                if not field_df.empty:
                    st.markdown(f"#### Field-level accuracy · largest completed sample ({largest_n} records)")
                    fig = px.bar(field_df, x="accuracy_pct", y="field", color="method", barmode="group", orientation="h", range_x=[0, 100], labels={"accuracy_pct": "Correct outputs (%)", "field": "Canonical field", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 620), use_container_width=True)

                with st.expander("Inspect model errors and missing/incorrect responses"):
                    errors = [call for call in st.session_state.llm_calls if call.get("error")]
                    if errors:
                        st.dataframe(pd.DataFrame([{k: call.get(k) for k in ["call_id", "timestamp_utc", "model_id", "record_keys", "elapsed_sec", "error"]} for call in errors]), use_container_width=True, hide_index=True)
                    else:
                        st.success("No API exceptions were logged.")
                    st.caption("A response that is valid JSON but has wrong or missing fields is still scored as incorrect for those fields.")

    with tab_audit:
        st.subheader("4 · Export reproducible evidence")
        st.write("Download the audit JSON after each successful run. Keep this export outside the public GitHub repository; it contains sampled source-derived records and experimental outputs.")
        has_benchmark = bool(st.session_state.gold_records)
        audit = make_audit_payload()
        st.download_button(
            "⬇️ Download complete audit trail (JSON)",
            data=json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            file_name="icitiit27_etl_experiment_audit.json",
            mime="application/json",
            disabled=not has_benchmark,
            use_container_width=True,
        )
        if has_benchmark:
            rows: list[dict[str, Any]] = []
            for key, gold in st.session_state.gold_records.items():
                rule_out = st.session_state.rule_outputs.get(key, {})
                llm_out = st.session_state.llm_outputs.get(key, {})
                item: dict[str, Any] = {
                    "record_key": key,
                    "gold_json": json.dumps(gold, ensure_ascii=False),
                    "corrupted_input_json": json.dumps(st.session_state.corrupted_records[key], ensure_ascii=False),
                    "rule_output_json": json.dumps(rule_out, ensure_ascii=False),
                    "llm_output_json": json.dumps(llm_out, ensure_ascii=False),
                    "rule_processing_seconds": st.session_state.rule_times.get(key),
                    "llm_call_id": st.session_state.llm_meta_by_key.get(key, {}).get("call_id"),
                }
                rows.append(item)
            st.download_button(
                "⬇️ Download record-level outputs (CSV)",
                data=pd.DataFrame(rows).to_csv(index=False),
                file_name="icitiit27_etl_record_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Build a benchmark before exporting record-level outputs.")

        st.markdown("#### Run metadata")
        metadata = audit["metadata"]
        meta_display = {k: metadata.get(k) for k in ["app_version", "research_title", "dataset_title", "dataset_source_url", "uploaded_filename", "uploaded_file_sha256", "uploaded_rows", "eligible_rows", "model_id", "random_seed", "benchmark_size", "benchmark_built_at_utc", "exported_at_utc"]}
        st.json(meta_display)
        st.warning("Never commit the API key, raw dataset, audit JSON or record-level CSV to a public GitHub repository. These files are experimental evidence and must be backed up privately.")

    with tab_protocol:
        st.subheader("5 · Research protocol and limitations")
        st.markdown("#### Experimental comparison")
        protocol_rows = [
            {"Component": "Dataset", "Protocol": f"Kaggle competition training file — [{DATASET_TITLE}]({DATA_SOURCE_URL})"},
            {"Component": "Analysis task", "Protocol": "Normalize controlled formatting and type variations; not return prediction"},
            {"Component": "Ground truth", "Protocol": "Canonicalized eligible input records captured before controlled corruption"},
            {"Component": "Rule-based system", "Protocol": "Explicit Python rules for numeric parsing, flags, case/whitespace and order-ID prefix"},
            {"Component": "LLM system", "Protocol": f"Groq API model `{model_id}`, temperature 0, JSON-mode response"},
            {"Component": "Primary accuracy metrics", "Protocol": "Field-level exact-within-tolerance accuracy and complete-record accuracy"},
            {"Component": "Efficiency metrics", "Protocol": "Measured processing wall time, records/second, token count and reference-priced service-equivalent cost"},
            {"Component": "Scale tests", "Protocol": "Only sample sizes actually run (20, 50, 100); no linear extrapolation"},
        ]
        st.dataframe(pd.DataFrame(protocol_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Limitations to state in the paper")
        st.markdown(
            "- The Kaggle listing describes transactions covering 2022–2024, but the original retailer and real-world provenance are not independently verified. Do not describe the data as confirmed small-business transactions.\n"
            "- This benchmark uses eligible source rows and adds controlled, reversible formatting/type variations. It does not test semantic correction of unknown errors, missing-value imputation, schema mapping between unrelated tables, or real production ingestion.\n"
            "- Because the transformation errors are known and reversible, a well-designed rule-based baseline is expected to perform strongly. The experiment is useful for testing when an LLM adds value, cost or latency in this bounded setting—not for assuming LLM superiority.\n"
            "- The reference cost excludes hosting, local compute, engineering time and any other operational expenses. Provider price/allowance must be verified on the experiment date.\n"
            "- A single sample run is exploratory. For a defensible paper, preserve audit files, record model/date/code version, repeat with predeclared seeds where practical, and report failures rather than silently discarding them."
        )
        st.markdown("#### Conference reference")
        st.markdown("ICITIIT 2027 author instructions: [official submission guidelines](https://icitiit27.iiitkottayam.ac.in/instructions-to-authors/). Confirm the current deadline, page limit and formatting requirements on the official site before submission.")


if __name__ == "__main__":
    main()
