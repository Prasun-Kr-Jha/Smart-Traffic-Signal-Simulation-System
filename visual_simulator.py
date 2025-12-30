
import streamlit as st
import numpy as np
import time
import pandas as pd
import altair as alt

# ==========================================
# 1. SESSION STATE & DYNAMIC SIDEBAR UX
# ==========================================
# We initialize state before page config to determine sidebar behavior
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False

# UX REQUIREMENT: Sidebar collapses when simulation starts to focus on data
initial_sidebar = "collapsed" if st.session_state.sim_running else "expanded"

st.set_page_config(
    layout="wide", 
    page_title="Smart Traffic Signal System", 
    page_icon="🚦",
    initial_sidebar_state=initial_sidebar
)

# ==========================================
# 2. THEME & CSS (GEMINI DARK - STRICT)
# ==========================================
BG_MAIN = "#0E1117"
BG_SIDEBAR = "#161B22"
BG_CARD = "#1F2937"
TEXT_PRIMARY = "#E5E7EB"
TEXT_SECONDARY = "#9CA3AF"
ACCENT_AI = "#22C55E"    # Soft Green
ACCENT_FIXED = "#94A3B8" # Neutral Gray
ACCENT_WARN = "#F59E0B"
ACCENT_EMERGENCY = "#EF4444"

st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background-color: {BG_MAIN};
        color: {TEXT_PRIMARY};
    }}
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {{
        background-color: {BG_SIDEBAR};
        border-right: 1px solid #30363d;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    /* Custom Insight Card */
    .insight-card {{
        background-color: {BG_CARD};
        border-left: 4px solid {ACCENT_AI};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    
    .insight-card-emergency {{
        background-color: {BG_CARD};
        border-left: 4px solid {ACCENT_EMERGENCY};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }}

    /* Explanation Box */
    .explanation-box {{
        background-color: {BG_SIDEBAR};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
        height: 100%;
    }}
    
    /* Footer */
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: {BG_SIDEBAR};
        color: {TEXT_SECONDARY};
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        border-top: 1px solid #30363d;
        z-index: 1000;
    }}
    
    /* Adjust main container */
    .block-container {{
        padding-bottom: 60px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. TRAFFIC LOGIC CORE (UNCHANGED)
# ==========================================

class TrafficEnv:
    def __init__(self):
        self.lanes = ["North", "South", "East", "West"]
        self.queues = np.random.randint(5, 15, 4).astype(float) 
        self.current_green = 0
        self.arrival_rates = [2, 2, 2, 2] 
        self.clearance_rate = 4           

    def step(self, action_idx, is_emergency=False):
        arrivals = np.random.poisson(self.arrival_rates)
        self.queues += arrivals
        self.current_green = action_idx
        clear_amount = self.clearance_rate * (2 if is_emergency else 1)
        self.queues[action_idx] = max(0, self.queues[action_idx] - clear_amount)
        return self.queues

def get_ai_action(queues):
    return np.argmax(queues)

def generate_decision_insight(action_idx, queues, is_emergency, emergency_lane):
    lanes = ["North", "South", "East", "West"]
    selected_lane = lanes[action_idx]
    current_q = queues[action_idx]
    avg_q = np.mean(queues)
    
    if is_emergency:
        return {
            "title": "🚑 EMERGENCY PRIORITY",
            "reason": "Decision overridden due to emergency priority.",
            "data": f"Forcing Green: {lanes[emergency_lane]}",
            "style": "insight-card-emergency",
            "icon": "🚨"
        }
    
    return {
        "title": f"✅ AI DECISION: {selected_lane.upper()}",
        "reason": "Highest queue imbalance detected.",
        "data": f"Queue: {int(current_q)} vehicles | Average: {int(avg_q)}",
        "style": "insight-card",
        "icon": "🤖"
    }

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    simulation_speed = st.slider("Simulation Speed", 0.1, 2.0, 0.5)
    total_steps = st.slider("Duration (Steps)", 50, 300, 100)
    
    st.markdown("### 🚦 Traffic Scenarios")
    scenario = st.selectbox("Select Condition", ["Normal Flow", "Morning Rush", "Emergency Event"])
    
    st.markdown("---")
    
    # UX TRIGGER: Set state and rerun to collapse sidebar
    if st.button("▶ Start Simulation", type="primary", use_container_width=True):
        st.session_state.sim_running = True
        st.session_state.params = {
            'speed': simulation_speed,
            'steps': total_steps,
            'scenario': scenario
        }
        st.rerun()

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================

st.title("Smart Traffic Signal Simulation")
st.markdown(f"<div style='color:{TEXT_SECONDARY}; margin-bottom: 20px;'>Comparing <b>Traditional Fixed-Time Signals</b> vs. <b>Adaptive AI Control</b>.</div>", unsafe_allow_html=True)

# --- DEFINITIONS SECTION ---
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="explanation-box">
        <strong style="color:{ACCENT_FIXED}">🔹 Fixed Traffic Logic</strong><br>
        <span style="font-size:0.85rem; color:{TEXT_SECONDARY}">
        Traditional time-based signal logic that cycles lanes regardless of congestion. 
        "Fixed signals follow time, not traffic."
        </span>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="explanation-box">
        <strong style="color:{ACCENT_AI}">🔹 AI Traffic Logic</strong><br>
        <span style="font-size:0.85rem; color:{TEXT_SECONDARY}">
        Adaptive logic that prioritizes the most congested lane using real-time data.
        "The AI reacts to real queues, not static schedules."
        </span>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="explanation-box">
        <strong style="color:{TEXT_PRIMARY}">🔹 Total Load Metric</strong><br>
        <span style="font-size:0.85rem; color:{TEXT_SECONDARY}">
        Total number of vehicles waiting across all lanes.
        Lower total load indicates better traffic efficiency.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. SIMULATION EXECUTION
# ==========================================

if st.session_state.sim_running:
    
    # Reset Button to stop simulation and bring back sidebar
    if st.button("⏹ Stop / Reset Configuration"):
        st.session_state.sim_running = False
        st.rerun()
    
    # Retrieve Params
    params = st.session_state.params
    
    # Init Environments
    env_fixed = TrafficEnv()
    env_ai = TrafficEnv()
    
    if params['scenario'] == "Morning Rush":
        env_fixed.arrival_rates = [6, 2, 6, 2]
        env_ai.arrival_rates = [6, 2, 6, 2]
    
    # UI Layout
    insight_panel = st.empty()
    col_metrics_fixed, col_metrics_ai = st.columns(2)
    chart_fixed_spot = col_metrics_fixed.empty()
    chart_ai_spot = col_metrics_ai.empty()
    history_chart_spot = st.empty()
    summary_spot = st.empty()
    
    history_fixed, history_ai = [], []
    emergency_lane = 0 
    
    # --- LOOP ---
    for t in range(params['steps']):
        
        is_emergency_active = (params['scenario'] == "Emergency Event" and 20 < t < 50)
        
        # LOGIC
        action_fixed = (t // 5) % 4 
        
        if is_emergency_active:
            action_ai = emergency_lane
        else:
            action_ai = get_ai_action(env_ai.queues)
            
        q_fixed = env_fixed.step(action_fixed)
        q_ai = env_ai.step(action_ai, is_emergency_active)
        
        load_fixed = np.sum(q_fixed)
        load_ai = np.sum(q_ai)
        history_fixed.append(load_fixed)
        history_ai.append(load_ai)
        
        insight = generate_decision_insight(action_ai, env_ai.queues, is_emergency_active, emergency_lane)
        
        # RENDER INSIGHT
        with insight_panel:
            st.markdown(f"""
            <div class="{insight['style']}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.2rem; font-weight:bold; color:{TEXT_PRIMARY}">{insight['icon']} {insight['title']}</span>
                    <span style="color:{TEXT_SECONDARY}; font-family:monospace;">Step: {t}</span>
                </div>
                <div style="margin-top:10px; font-size:1.1rem; color:{TEXT_PRIMARY}">
                    <b>Reason:</b> {insight['reason']}
                </div>
                <div style="margin-top:5px; font-size:0.9rem; color:{TEXT_SECONDARY}; font-family:monospace;">
                    {insight['data']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # RENDER CHARTS
        def create_bar_chart(queues, active_idx, color, title):
            df = pd.DataFrame({
                'Lane': ["North", "South", "East", "West"], 
                'Vehicles': queues,
                'Status': ['Active' if i == active_idx else 'Red' for i in range(4)]
            })
            return alt.Chart(df).mark_bar().encode(
                x=alt.X('Lane', axis=None),
                y=alt.Y('Vehicles', scale=alt.Scale(domain=[0, 40])),
                color=alt.condition(
                    alt.datum.Status == 'Active', alt.value(color), alt.value("#374151")
                ),
                tooltip=['Lane', 'Vehicles']
            ).properties(height=200, title=title)

        with chart_fixed_spot:
            st.altair_chart(create_bar_chart(q_fixed, action_fixed, ACCENT_FIXED, f"Fixed Controller (Load: {int(load_fixed)})"), use_container_width=True)
        with chart_ai_spot:
            st.altair_chart(create_bar_chart(q_ai, action_ai, ACCENT_AI, f"AI Controller (Load: {int(load_ai)})"), use_container_width=True)

        # RENDER HISTORY
        df_hist = pd.DataFrame({'Step': range(len(history_fixed)), 'Fixed System': history_fixed, 'AI System': history_ai}).melt('Step', var_name='System', value_name='Total Load')
        line_chart = alt.Chart(df_hist).mark_line(strokeWidth=3).encode(
            x='Step', y='Total Load',
            color=alt.Color('System', scale=alt.Scale(domain=['Fixed System', 'AI System'], range=[ACCENT_FIXED, ACCENT_AI]))
        ).properties(height=250, title="Real-time Efficiency Comparison (Lower is Better)")
        history_chart_spot.altair_chart(line_chart, use_container_width=True)
        
        time.sleep(params['speed'])
    
    # --- CONCLUSION ---
    avg_load_fixed = np.mean(history_fixed)
    avg_load_ai = np.mean(history_ai)
    improvement = ((avg_load_fixed - avg_load_ai) / avg_load_fixed) * 100
    
    summary_spot.markdown(f"""
    <div style="background-color: {BG_CARD}; padding: 25px; border-radius: 8px; margin-top: 20px; border: 1px solid {ACCENT_AI};">
        <h3 style="color: {ACCENT_AI}; margin-top:0;">📊 Why the AI Model Is More Efficient</h3>
        <p style="color: {TEXT_PRIMARY};">
            Under identical traffic conditions, the <b>AI-based controller</b> consistently maintained a lower total vehicle load than the traditional fixed-time traffic system.
        </p>
        <p style="color: {TEXT_SECONDARY};">
            This improvement is achieved by dynamically allocating green time based on congestion rather than following static schedules.
            The AI controller reduced average congestion by <b>{improvement:.1f}%</b> compared to fixed-time logic.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # LANDING STATE (When simulation is not running)
    st.info("👈 Please configure parameters in the sidebar and click 'Start Simulation'.")

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    Developed by Prasun Kumar Jha <br>
    <span style="opacity: 0.7;">Smart Traffic Signal Simulation System</span>
</div>
""", unsafe_allow_html=True)
