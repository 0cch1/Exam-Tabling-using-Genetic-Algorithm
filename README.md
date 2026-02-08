# Exam Timetabling using Genetic Algorithms

This repository contains an implementation of a Genetic Algorithm (GA) for solving
the exam timetabling problem as part of the CT421 Artificial Intelligence course.

## Problem Description

The goal is to assign exams to timeslots such that:
- **Hard constraint**: No student has two exams scheduled in the same timeslot.
- **Soft constraint**: The number of consecutive exams taken by students is minimized.

## Repository Structure

- **src/** — GA implementation
- **instances/** — Problem instances (small, medium, test case)
- **experiments/** — Scripts to reproduce experiments
- **results/** — Output summaries and plots

## Requirements

- Python 3.9+
- numpy
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run Experiments

**Small instance**
```bash
python experiments/run_small.py
```

**Medium instance**
```bash
python experiments/run_medium.py
```

**Test case 1**
```bash
python experiments/run_test_case1.py
```

Each script runs multiple independent GA executions with different random seeds
and reports best, mean, worst, and standard deviation of the soft constraint cost.
