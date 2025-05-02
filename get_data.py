import streamlit as st
from datetime import datetime
import pandas as pd

st.markdown("""
<style>
.task-card {
    border-left: 4px solid #4CAF50;
    border-radius: 4px;
    padding: 16px;
    margin: 8px 0;
    background-color: transparent;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.task-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.task-card h3 {
    margin-top: 0;
    color: var(--text-color);
}
.task-card p {
    margin-bottom: 0;
    color: var(--text-color-secondary);
}
button[title] {
    font-size: 20px;
    background: none;
    border: none;
    cursor: pointer;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


def main():
    st.title("📝 To-Do List App")
    st.write("Organize your tasks with style!")
    
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    
    # Sidebar for adding tasks
    with st.sidebar:
        st.header("Add New Task")
        new_task = st.text_input("Task description")
        due_date = st.date_input("Due date")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        
        if st.button("Add Task"):
            if new_task:
                task = {
                    "id": len(st.session_state.tasks) + 1,
                    "description": new_task,
                    "due_date": due_date,
                    "priority": priority,
                    "created_at": datetime.now(),
                    "completed": False
                }
                st.session_state.tasks.append(task)
                st.success("Task added successfully!")
            else:
                st.warning("Please enter a task description")
    
    st.header("Your Tasks")
    
    if not st.session_state.tasks:
        st.info("No tasks yet. Add some using the sidebar!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            show_completed = st.checkbox("Show completed tasks")
        with col2:
            sort_option = st.selectbox("Sort by", ["Creation Date", "Due Date", "Priority"])
        
        filtered_tasks = [task for task in st.session_state.tasks if show_completed or not task["completed"]]
        
        if sort_option == "Due Date":
            filtered_tasks.sort(key=lambda x: x["due_date"])
        elif sort_option == "Priority":
            order = {"High": 1, "Medium": 2, "Low": 3}
            filtered_tasks.sort(key=lambda x: order[x["priority"]])
        else:
            filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)

        for task in filtered_tasks:
            border_color = {
                "High": "#ff4b4b",
                "Medium": "#ffa500",
                "Low": "#4CAF50"
            }.get(task["priority"], "#4CAF50")

            with st.form(key=f"form_{task['id']}"):
                st.markdown(
                    f"""
                    <div class="task-card" style="border-left-color: {border_color}; {'opacity: 0.6;' if task['completed'] else ''}">
                        <h3>{task['description']}</h3>
                        <p>📅 Due: {task['due_date'].strftime('%b %d, %Y')}</p>
                        <p>🔖 Priority: {task['priority']}</p>
                        <p>⏱ Created: {task['created_at'].strftime('%b %d, %Y %H:%M')}</p>
                        <div style="margin-top: 10px;">
                    """, unsafe_allow_html=True
                )

                colA, colB = st.columns([1, 1])
                with colA:
                    complete = st.form_submit_button("✅", help="Mark Complete")
                with colB:
                    delete = st.form_submit_button("🗑️", help="Delete Task")

                if complete:
                    task["completed"] = not task["completed"]
                    st.rerun()
                if delete:
                    st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task["id"]]
                    st.rerun()

                st.markdown("</div></div>", unsafe_allow_html=True)
    
        # Exportable dataset section
    task_df = pd.DataFrame(st.session_state.tasks)

    if not task_df.empty:
        # Convert datetime objects to strings for CSV export
        task_df["created_at"] = task_df["created_at"].astype(str)
        task_df["due_date"] = task_df["due_date"].astype(str)
        
        st.subheader("📊 Export Task Data")
        st.write("Use this for productivity prediction or analysis.")

        # Download button
        csv = task_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Task Dataset as CSV",
            data=csv,
            file_name="tasks_dataset.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
