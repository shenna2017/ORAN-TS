# HyperTriGNN and TriGNN Models for 5G/6G ORAN Data

This project implements hypergraph neural networks, specifically the HyperTriGNN and TriGNN models, to analyze and predict Channel Quality Indicators (CQI) using ORAN data for traffic steering. The models leverage the complex relationships within hypergraphs to enhance prediction accuracy for network traffic steering and resource management.

## Overview

The HyperTriGNN model incorporates hypergraph structures to capture the relationships between multiple nodes simultaneously, making it particularly effective for modeling CQI from network data. In contrast, the TriGNN model operates on traditional graph representations, offering a more straightforward approach to graph neural networks.

## Dataset Details

The dataset used for this project is sourced from:

**Berkay Koksal, Robert Schmidt, Xenofon Vasilakos, Navid Nikaien.**  
**CRAWDAD dataset:** `eurecom/elasticmon5G2019` (v. 2019‑08‑29), traceset: `02‑PreprocessedDatasets`.  
**Download Link:** [CRAWDAD - eurecom/elasticmon5G2019](https://crawdad.org/eurecom/elasticmon5G2019/20190829/02‑PreprocessedDatasets)

This dataset was downloaded in August 2019 and contains preprocessed measurements essential for modeling CQI predictions in 5G/LTE networks.

## Installation

To run the models, ensure you have the following libraries installed:

```bash
pip install dgl torch pandas numpy scikit-learn
# ORAN-TS
