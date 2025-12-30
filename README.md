🚦 Smart Traffic Signal Simulation System
Explainable AI vs Traditional Traffic Control
📌 Overview

This project demonstrates why an AI-based traffic signal controller is more efficient than a traditional fixed-time traffic light, using a real-time simulation with clear visual comparison and explainable decisions.

Instead of following static schedules, the AI controller adapts signal priorities based on live traffic conditions, reducing congestion and improving overall traffic flow.

The system is designed to be:

Explainable (no black-box decisions)

Visually intuitive

Technically honest

Easy to understand for non-experts

🎯 Problem Statement

Most real-world traffic signals operate on fixed time cycles:

Each lane gets green for a predefined duration

Signals do not react to congestion

Empty lanes may receive green while congested lanes wait

This leads to:

Increased waiting time

Higher congestion

Poor handling of traffic imbalance and emergencies

💡 Solution Approach

The project compares two traffic control strategies under identical conditions:

1️⃣ Fixed-Time Traffic Logic (Baseline)

Cycles through lanes sequentially

Uses predefined timing

Does not observe real-time traffic

Represents traditional traffic systems

2️⃣ AI-Based Adaptive Traffic Logic

Observes live traffic queues

Prioritizes congested lanes

Adjusts decisions dynamically

Overrides logic during emergency conditions

The comparison is performed side-by-side, ensuring fairness and clarity.

🧠 Key Concept: Explainable AI (Level-2)

This project intentionally focuses on explainability, not complexity.

Each AI decision is accompanied by a Decision Insight Panel that explains:

Which lane was selected

Why it was selected

How its congestion compares to other lanes

During emergency conditions, the system clearly states:

Decision overridden due to emergency priority.

This ensures the AI is transparent and understandable, not a black box.

📊 Core Metrics Explained
🔹 Total Load

Definition:
Total number of vehicles waiting across all lanes.

Why it matters:
Higher total load means more congestion and longer delays.

Interpretation:
Lower total load = better traffic efficiency.

🔹 Fixed Load

Total load under traditional fixed-time signal logic.

🔹 AI Load

Total load under AI-based adaptive control.

🚑 Emergency Mode

The system includes an emergency priority mode to simulate real-world requirements.

Behavior:

Emergency activates during a defined time window

AI decision logic is overridden

A specific lane is prioritized

Clearance rate is increased

Decision explanation switches to emergency notice

This models realistic emergency handling in traffic systems.

🎨 User Interface Design

The application uses a Gemini-style dark theme:

Calm, readable, and professional

High contrast for clarity

No flashy animations or distractions

Sidebar Behavior:

Used for configuration (speed, steps, scenario, emergency)

Automatically collapses after simulation starts

Remains accessible if adjustments are needed

This keeps the focus on analysis and results.

📈 Why the AI Controller Is More Efficient

Under identical traffic conditions:

The fixed-time controller distributes green time evenly, regardless of congestion.

The AI controller allocates green time where congestion is highest.

As a result:

Congested lanes are cleared faster

Waiting time is reduced

Overall traffic flow improves

The improvement is visible in:

Lower total load values

More balanced lane queues

Smoother congestion trends over time

🛠️ Technology Stack

Python

Streamlit – Interactive UI

NumPy – Traffic simulation logic

Altair – Real-time visualizations

Reinforcement Learning (DQN) – Adaptive control logic

⚠️ Scope & Limitations

This project intentionally focuses on:

A single intersection

Explainability over complexity

Demonstration rather than city-scale deployment

It does not include:

Multi-intersection coordination

Advanced statistical modeling

Real-world traffic datasets

These are considered future extensions.

🚀 How to Run
pip install -r requirements.txt
streamlit run app.py

🧾 Credits

Developed by Prasun
Smart Traffic Signal Simulation System

🏁 Final Note

This project shows how adaptive, explainable AI can outperform traditional infrastructure logic — not by being complex, but by being responsive to real conditions.

Fixed signals follow time.
Intelligent signals respond to reality.