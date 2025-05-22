import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

def analyze_productivity(tasks):
    """Analyze productivity using ML-based predictions"""
    if not tasks:
        return None, None
    
    try:
        df = pd.DataFrame(tasks)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['due_date'] = pd.to_datetime(df['due_date'])
        
        # Calculate time-based features
        df['days_to_due'] = (df['due_date'] - df['created_at']).dt.days
        df['is_completed'] = df['completed'].astype(int)
        
        # Encode priority
        le = LabelEncoder()
        df['priority_encoded'] = le.fit_transform(df['priority'])
        
        # Prepare features for ML
        features = ['days_to_due', 'priority_encoded']
        X = df[features]
        y = df['is_completed']
        
        # Train a simple ML model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate completion probability for each task
        df['completion_probability'] = model.predict_proba(X)[:, 1]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Priority completion rate
        priority_stats = df.groupby('priority')['is_completed'].mean()
        priority_stats.plot(kind='bar', ax=ax1, color=['#ff4b4b', '#ffa500', '#4CAF50'])
        ax1.set_title('Completion Rate by Priority')
        ax1.set_ylabel('Completion Rate')
        
        # Days to due vs completion probability
        df.plot(kind='scatter', x='days_to_due', y='completion_probability', ax=ax2)
        ax2.set_title('Completion Probability vs Days to Due')
        ax2.set_xlabel('Days to Due Date')
        ax2.set_ylabel('Completion Probability')
        
        plt.tight_layout()
        
        return df, fig
    except Exception as e:
        st.error(f"Error in productivity analysis: {str(e)}")
        return None, None

def analyze_daily_productivity(tasks):
    """Analyze productivity for the current day and predict completion"""
    today = datetime.now().date()
    today_tasks = [task for task in tasks if task['due_date'] == today]
    
    if not today_tasks:
        return None, None, None
    
    # Calculate completion rate for today
    completed_today = sum(1 for task in today_tasks if task['completed'])
    total_today = len(today_tasks)
    completion_rate = (completed_today / total_today) * 100 if total_today > 0 else 0
    
    # Estimate time needed for remaining tasks
    avg_task_time = 30  # Assuming average task takes 30 minutes
    remaining_tasks = total_today - completed_today
    estimated_time_needed = remaining_tasks * avg_task_time
    
    # Calculate remaining hours in the day
    current_hour = datetime.now().hour
    remaining_hours = 24 - current_hour
    
    # Predict if tasks can be completed today
    can_complete = estimated_time_needed <= (remaining_hours * 60)
    
    return today_tasks, completion_rate, can_complete

def generate_timetable(tasks):
    """Generate a timetable for the day"""
    today = datetime.now().date()
    today_tasks = [task for task in tasks if task['due_date'] == today and not task['completed']]
    
    if not today_tasks:
        return None
    
    # Sort tasks by priority
    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    today_tasks.sort(key=lambda x: priority_order[x["priority"]])
    
    # Generate timetable
    current_time = datetime.now()
    timetable = []
    
    for task in today_tasks:
        # Allocate time based on priority
        time_allocation = {
            "High": 60,  # 1 hour
            "Medium": 45,  # 45 minutes
            "Low": 30  # 30 minutes
        }
        
        task_time = time_allocation[task["priority"]]
        start_time = current_time
        end_time = start_time + timedelta(minutes=task_time)
        
        timetable.append({
            "task": task["description"],
            "priority": task["priority"],
            "start_time": start_time,
            "end_time": end_time,
            "duration": task_time
        })
        
        current_time = end_time
    
    return timetable

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
    
    # Create tabs for tasks and analysis
    tab1, tab2 = st.tabs(["📋 Tasks", "📊 Productivity Analysis"])
    
    with tab1:
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
                task_df["created_at"] = task_df["created_at"].astype(str)
                task_df["due_date"] = task_df["due_date"].astype(str)
                
                st.subheader("📊 Export Task Data")
                st.write("Use this for productivity prediction or analysis.")

                csv = task_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Task Dataset as CSV",
                    data=csv,
                    file_name="tasks_dataset.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.header("Productivity Analysis")
        if len(st.session_state.tasks) > 0:
            # Daily Productivity Analysis
            st.subheader("📅 Today's Productivity")
            today_tasks, completion_rate, can_complete = analyze_daily_productivity(st.session_state.tasks)
            
            if today_tasks:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Today's Tasks", len(today_tasks))
                with col2:
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                with col3:
                    status = "✅ Can Complete" if can_complete else "⚠️ May Need More Time"
                    st.metric("Today's Status", status)
                
                # Generate and display timetable
                st.subheader("📋 Today's Timetable")
                timetable = generate_timetable(st.session_state.tasks)
                
                if timetable:
                    for slot in timetable:
                        with st.expander(f"{slot['start_time'].strftime('%I:%M %p')} - {slot['end_time'].strftime('%I:%M %p')} | {slot['task']}"):
                            st.write(f"**Priority:** {slot['priority']}")
                            st.write(f"**Duration:** {slot['duration']} minutes")
                            st.write(f"**Task:** {slot['task']}")
                else:
                    st.info("No remaining tasks for today!")
            
            # Original ML Analysis
            df, fig = analyze_productivity(st.session_state.tasks)
            if df is not None:
                st.subheader("ML-Powered Insights")
                
                # Calculate key metrics
                total_tasks = len(df)
                completed_tasks = df['is_completed'].sum()
                completion_rate = (completed_tasks / total_tasks) * 100
                high_risk_tasks = df[df['completion_probability'] < 0.3]
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tasks", total_tasks)
                with col2:
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                with col3:
                    st.metric("High Risk Tasks", len(high_risk_tasks))
                
                # Display visualizations
                st.pyplot(fig)
                
                # Display recommendations
                st.subheader("Recommendations")
                if len(high_risk_tasks) > 0:
                    st.warning(f"⚠️ You have {len(high_risk_tasks)} high-risk tasks that need attention!")
                    for _, task in high_risk_tasks.iterrows():
                        st.write(f"- {task['description']} (Priority: {task['priority']})")
                
                # Priority-based insights with improved styling
                st.subheader("Priority Analysis")
                priority_stats = df.groupby('priority').agg({
                    'is_completed': ['count', 'mean'],
                    'completion_probability': 'mean'
                }).round(2)
                
                # Rename columns for better readability
                priority_stats.columns = ['Task Count', 'Completion Rate', 'Predicted Success Rate']
                
                # Style the dataframe
                styled_stats = priority_stats.style.format({
                    'Completion Rate': '{:.1%}',
                    'Predicted Success Rate': '{:.1%}'
                }).background_gradient(
                    subset=['Completion Rate', 'Predicted Success Rate'],
                    cmap='RdYlGn'
                ).set_properties(**{
                    'background-color': 'white',
                    'color': 'black',
                    'border-color': 'lightgrey',
                    'border-style': 'solid',
                    'border-width': '1px',
                    'padding': '5px'
                })
                
                st.dataframe(styled_stats, use_container_width=True)
                
                # Add explanation
                st.info("""
                📊 **Understanding the Analysis:**
                - **Task Count**: Number of tasks in each priority level
                - **Completion Rate**: Historical completion rate for each priority
                - **Predicted Success Rate**: ML-predicted probability of completion
                """)
        else:
            st.info("Add some tasks to see productivity analysis!")

if __name__ == "__main__":
    main()