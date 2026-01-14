# 🚗 Collaborative Reservation and Path Guidance Strategy for Electric Vehicle Charging Stations Oriented to Dynamic Traffic Flow

> **A "Vehicle-Road-Station" Collaborative Scheduling System based on LSTM Traffic Prediction and M/M/N Queuing Theory.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 📖 Abstract

With the rapid growth of Electric Vehicles (EVs), the structural contradiction of "more cars, fewer piles" and the uneven spatiotemporal distribution of charging resources have led to severe queuing and localized traffic congestion.

Unlike traditional static navigation, this project proposes a **Collaborative Scheduling System**. It integrates:
1.  **Dynamic Perception**: Utilizing **LSTM** neural networks to predict short-term traffic flow in urban networks.
2.  **State Assessment**: Using **M/M/N Queuing Theory** to precisely quantify the waiting cost at charging stations.
3.  **Collaborative Decision**: Employing improved **Genetic Algorithms (GA / NSGA-II)** for global optimal scheduling.

**🎯 Core Objective**: To minimize the user's total travel cost (Travel Time + Waiting Time) while maximizing the load balance of the charging network (Peak Shaving and Valley Filling).


# 🚗 Collaborative Reservation and Path Guidance Strategy for Electric Vehicle Charging Stations Oriented to Dynamic Traffic Flow

> **A "Vehicle-Road-Station" Collaborative Scheduling System based on LSTM Traffic Prediction and M/M/N Queuing Theory.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 📖 Abstract

With the rapid growth of Electric Vehicles (EVs), the structural contradiction of "more cars, fewer piles" and the uneven spatiotemporal distribution of charging resources have led to severe queuing and localized traffic congestion.

Unlike traditional static navigation, this project proposes a **Collaborative Scheduling System**. It integrates:
1.  **Dynamic Perception**: Utilizing **LSTM** neural networks to predict short-term traffic flow in urban networks.
2.  **State Assessment**: Using **M/M/N Queuing Theory** to precisely quantify the waiting cost at charging stations.
3.  **Collaborative Decision**: Employing improved **Genetic Algorithms (GA / NSGA-II)** for global optimal scheduling.

**🎯 Core Objective**: To minimize the user's total travel cost (Travel Time + Waiting Time) while maximizing the load balance of the charging network (Peak Shaving and Valley Filling).


### Module 3: Project Structure & Core Modules

```markdown
## 📂 Project Structure

The project repository is organized as follows:

```text
📁 battery_swapping_strategy
├── 📂 TrafficFlowPrediction-master  # [Module 1] Traffic Prediction
│   ├── data/                        # Training datasets
│   ├── model/                       # LSTM/GRU model weights (.h5)
│   └── main.py                      # Prediction entry point
├── 📂 group                         # [Module 2] Collaborative Optimization (Core)
│   ├── c101_21.xlsx - Nodes.csv     # Network Node Data (Modified Solomon C101)
│   ├── group_opt.py                 # Main GA Optimization Script
│   ├── nsga2_principle_diagram.png  # Algorithm schematic
│   └── ga_final_result.csv          # Output: Final scheduling plan
├── 📂 queuing theory                # [Module 3] Queuing Cost Validation
│   ├── min_cost.py                  # Cost calculation logic
│   └── balance_cost.py              # Load balance metrics
├── 📂 min_cost                      # Baseline: Single-user Greedy Strategy
└── 📂 slides                        # Academic Reports & Presentations