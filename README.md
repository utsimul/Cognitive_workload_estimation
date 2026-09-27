# Cognitive_workload_estimation

Working on two task definitions:

Definition 1:
To predict task or trial type (such as arithmetic easy, arithmetic hard, stroop easy, etc) using physiological signals

Definition 2:
To predict cognitive load level based on physiological signals. The congitive load can be defined by us - as a continuous float value or discrete cognitive load levels. Actual values for these can be calculated based on the labels given in the EPIStress dataset.

Input ($x_{phys}$):
• EEG band powers
• EDA features
• HRV features
• TEMP features

The dataset used is EPIStress dataset.
Citation:
S. Moontaha, C. Cavalier, B. Esser, A. Jordan, I. Goebel, C. Anders,
A. Mimi, B. Kr¨uger, R. Surges, and B. Arnrich, “Epistress: A multimodal
dataset of physiological signals to measure cognitive stress in epilepsy
patients,” Scientific Data, 2025

#### Comparing the Preprocessed and Features data inside EPIStress:

| Signal | Preprocessed shape | Samples/window | Feature shape | Features/window |
| --- | --- | --- | --- | --- |
| **EEG** | `(154368, 4)` | `154368 / 96 = 1608` | `(96, 193)` | 193 |
| **EDA** | `(2412,)` | `2412 / 96 = 25.125` | `(96, 6)` | 6 |
| **TEMP** | `(2412,)` | `2412 / 96 = 25.125` | `(96, 2)` | 2 |
| **PPG/BVP** | `(38592,)` | `38592 / 96 = 402` | `(96, 5)` | 5 |

#### Feature vector size:

Unprocessed feature vector consists of 4 types of physiological data:
1. EEG: (96, 193)
2. EDA: (96, 6)
3. PPG: (96, 5)
4. TEMP: (96, 2)
So the unprocessed feature vector size (without EDA) can be written as (96, 206) if we concatenate them. 

#### Approach 1 (for EDA and concatenation):

Average each feature over 96 rows so that we have one value for every feature for every task (for every patient). Medical relevance to be tested.

(`approach_avg` folder)