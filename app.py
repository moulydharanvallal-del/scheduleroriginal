import json
import streamlit as st

from scheduler_core import (
    run_scheduler,
    bom_data as DEFAULT_BOM,
    customer_orders as DEFAULT_ORDERS,
    work_center_capacity as DEFAULT_CAPACITY,
)

st.set_page_config(page_title="Mini Manufacturing Scheduler", layout="wide")

st.title("Mini Manufacturing Scheduler")
st.caption("Paste/edit JSON inputs, run the scheduler, and share the app with others.")

with st.sidebar:
    st.header("Run")
    show_chart = st.checkbox("Show Gantt chart", value=True)
    run = st.button("Run scheduler", type="primary")

tab1, tab2, tab3 = st.tabs(["Inputs", "Results", "How to share"])

with tab1:
    st.subheader("Customer orders (JSON list)")
    orders_text = st.text_area(
        "orders_json",
        value=json.dumps(DEFAULT_ORDERS, indent=2),
        height=240,
        label_visibility="collapsed",
    )

    st.subheader("BOM / routing data (JSON list)")
    bom_text = st.text_area(
        "bom_json",
        value=json.dumps(DEFAULT_BOM, indent=2),
        height=320,
        label_visibility="collapsed",
    )

    st.subheader("Work-center capacity (JSON object)")
    cap_text = st.text_area(
        "capacity_json",
        value=json.dumps(DEFAULT_CAPACITY, indent=2),
        height=220,
        label_visibility="collapsed",
    )

    st.info("Tip: keep keys/fields consistent. If JSON is invalid, the run will fail with a helpful error.")

def _parse_json(name, txt):
    try:
        return json.loads(txt)
    except Exception as e:
        raise ValueError(f"{name} JSON error: {e}")

if run:
    try:
        orders = _parse_json("Orders", orders_text)
        bom = _parse_json("BOM", bom_text)
        capacity = _parse_json("Capacity", cap_text)

        with st.spinner("Scheduling..."):
            scheduled, work_orders, plan, fig = run_scheduler(
                bom, orders, capacity, show_chart=show_chart
            )

        st.session_state["scheduled"] = scheduled
        st.session_state["work_orders"] = work_orders
        st.session_state["plan"] = plan
        st.session_state["fig"] = fig
        st.success(f"Done. Scheduled runs: {len(scheduled)} | Work orders: {len(work_orders)}")

    except Exception as e:
        st.error(str(e))

with tab2:
    scheduled = st.session_state.get("scheduled")
    work_orders = st.session_state.get("work_orders")
    plan = st.session_state.get("plan")
    fig = st.session_state.get("fig")

    if not scheduled:
        st.warning("Run the scheduler from the sidebar to see results.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Scheduled rows", len(scheduled))
        c2.metric("Work orders", len(work_orders) if work_orders else 0)
        c3.metric("Ledger rows", len(plan.get("ledger", [])) if plan else 0)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scheduled table")
        st.dataframe(scheduled, use_container_width=True, height=320)

        with st.expander("Work orders"):
            st.dataframe(work_orders, use_container_width=True, height=260)

        with st.expander("Plan ledger"):
            st.dataframe(plan.get("ledger", []), use_container_width=True, height=260)

with tab3:
    st.subheader("Share options (lightweight)")
    st.markdown(
        """
**Option A (easiest): Streamlit Community Cloud**
1. Put these files in a GitHub repo
2. In Streamlit Cloud, deploy `app.py`
3. Share the URL

**Option B (internal): run locally**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Option C (single binary): PyInstaller**
If you *need* a no-Python install, you can bundle a binary (larger download):
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole app.py
```
        """
    )
