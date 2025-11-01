# Conceptual Diagrams for Statistical Methodology Framework

## Figure 1: Multi-Scale Temporal Hierarchy

```mermaid
graph TD
    A[Raw Weather Data<br/>Hourly Resolution] --> B[Level 1: Instantaneous WSI<br/>Point-wise Features]
    B --> C[Level 2: Rolling 6-hour Window<br/>Temporal Context]
    C --> D[Level 3: Temporal Smoothing<br/>HMM/Median Filter]
    D --> E[Level 4: Stability Labels<br/>Stable/Unstable]
    
    F[Weather Features:<br/>• Temperature<br/>• Pressure<br/>• Wind<br/>• Precipitation<br/>• Cloudiness] --> B
    
    G[Window Statistics:<br/>• Mean WSI<br/>• Standard Deviation<br/>• Trend Slope] --> C
    
    H[Smoothing Methods:<br/>• Hidden Markov Model<br/>• Median Filter<br/>• EWMA] --> D
    
    I[Final Classification:<br/>• Binary Labels<br/>• Probability Scores<br/>• Regime Duration] --> E
    
    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
    style I fill:#fff3e0
```

## Figure 2: Hidden Markov Model State Transitions

```mermaid
graph LR
    S1[Stable State<br/>S₁] --> S2[Unstable State<br/>S₂]
    S2 --> S1
    S1 --> S1
    S2 --> S2
    
    S1 -.-> O1[Observations:<br/>Low WSI<br/>Low Variability<br/>Stable Trends]
    S2 -.-> O2[Observations:<br/>High WSI<br/>High Variability<br/>Rapid Changes]
    
    T1[Transition Probabilities:<br/>P(S₂|S₁) = 0.1<br/>P(S₁|S₂) = 0.3<br/>P(S₁|S₁) = 0.9<br/>P(S₂|S₂) = 0.7]
    
    E1[Emission Probabilities:<br/>P(O|S₁) ~ N(μ₁, σ₁²)<br/>P(O|S₂) ~ N(μ₂, σ₂²)]
    
    style S1 fill:#c8e6c9
    style S2 fill:#ffcdd2
    style O1 fill:#e8f5e8
    style O2 fill:#ffe8e8
    style T1 fill:#fff3e0
    style E1 fill:#fff3e0
```

## Figure 3: Validation Pipeline Flowchart

```mermaid
flowchart TD
    A[Raw Weather Data<br/>Hourly, 2024 Germany] --> B[Feature Engineering<br/>11 Weather Attributes]
    B --> C[WSI Computation<br/>Robust Normalization]
    C --> D[Temporal Aggregation<br/>6-hour Windows]
    D --> E[GMM Classification<br/>Stability Regimes]
    E --> F[HMM Smoothing<br/>Temporal Dependence]
    F --> G[Stability Labels<br/>Stable/Unstable Timeline]
    
    H[Renewable Energy Data<br/>Solar & Wind Production] --> I[Model Predictions<br/>ARIMA, Prophet, Persistence]
    I --> J[Performance Metrics<br/>MAE, RMSE, MAPE]
    
    G --> K[Merge Datasets<br/>Stability + Performance]
    J --> K
    K --> L[Statistical Testing<br/>Mann-Whitney U, t-tests]
    L --> M[Effect Size Calculation<br/>Cohen's d, % Increase]
    M --> N[Robustness Metrics<br/>RPD, APD, RI]
    N --> O[Model Rankings<br/>Accuracy + Robustness]
    
    P[Validation Steps:<br/>• Cross-validation<br/>• Bootstrap CI<br/>• Permutation tests<br/>• Sensitivity analysis] --> L
    
    Q[Quality Checks:<br/>• Silhouette score<br/>• ACF analysis<br/>• Regime duration<br/>• Power analysis] --> E
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
    style O fill:#ffecb3
    style P fill:#fff3e0
    style Q fill:#fff3e0
```

## Figure 4: Statistical Test Framework

```mermaid
graph TD
    A[Model Performance Data<br/>MAE by Stability Regime] --> B[Test 1: Mann-Whitney U<br/>Non-parametric]
    A --> C[Test 2: Welch's t-test<br/>Parametric]
    A --> D[Test 3: Mixed-Effects Model<br/>Temporal Clustering]
    A --> E[Test 4: Permutation Test<br/>Distribution-free]
    
    B --> F[Effect Sizes:<br/>• Rank-biserial correlation<br/>• Cohen's d<br/>• Percentage increase]
    C --> F
    D --> F
    E --> F
    
    F --> G[Multiple Testing Correction<br/>Bonferroni or FDR]
    G --> H[Significance Decision<br/>p < α_corrected]
    
    I[Validation Metrics:<br/>• Confidence intervals<br/>• Bootstrap validation<br/>• Cross-validation<br/>• Sensitivity analysis] --> H
    
    J[Practical Significance:<br/>• ≥15% error increase<br/>• Effect size ≥0.3<br/>• Operational impact] --> H
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style F fill:#fff3e0
    style I fill:#fff3e0
    style J fill:#fff3e0
```

## Figure 5: Robustness Metrics Comparison

```mermaid
graph LR
    A[Model Performance Data] --> B[Metric 1: RPD<br/>Relative Performance Degradation<br/>Primary Metric]
    A --> C[Metric 2: APD<br/>Absolute Performance Degradation<br/>Supplementary]
    A --> D[Metric 3: RI<br/>Robustness Index<br/>Composite Score]
    A --> E[Metric 4: CV<br/>Coefficient of Variation<br/>Tie-breaker]
    A --> F[Metric 5: Skill Preservation<br/>Skill Score Ratio<br/>Context-dependent]
    
    B --> G[Primary Ranking<br/>Lowest RPD = Most Robust]
    C --> H[Secondary Filter<br/>Absolute Impact Assessment]
    D --> I[Combined Score<br/>Accuracy + Robustness]
    E --> J[Consistency Check<br/>Performance Stability]
    F --> K[Baseline Comparison<br/>Persistence Model]
    
    G --> L[Final Model Rankings<br/>1. Primary: RPD<br/>2. Filter: Accuracy threshold<br/>3. Tie-breaker: CV]
    H --> L
    I --> L
    J --> L
    K --> L
    
    M[Statistical Validation:<br/>• Bootstrap CI<br/>• Ranking uncertainty<br/>• Significance tests] --> L
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style B fill:#ffecb3
    style M fill:#fff3e0
```

## Figure 6: Implementation Architecture

```mermaid
graph TD
    A[Data Input<br/>Weather + Energy Data] --> B[Preprocessing<br/>Timestamp Alignment<br/>Missing Data Handling]
    B --> C[Feature Engineering<br/>WSI Computation<br/>Rolling Statistics]
    C --> D[Stability Classification<br/>GMM + HMM]
    D --> E[Model Evaluation<br/>Performance Metrics]
    E --> F[Statistical Analysis<br/>Hypothesis Testing]
    F --> G[Robustness Assessment<br/>Model Rankings]
    G --> H[Results Output<br/>Reports + Visualizations]
    
    I[Configuration<br/>parameters.yaml] --> B
    I --> C
    I --> D
    I --> E
    I --> F
    I --> G
    
    J[Validation Framework<br/>Cross-validation<br/>Bootstrap<br/>Sensitivity Analysis] --> D
    J --> E
    J --> F
    J --> G
    
    K[Quality Assurance<br/>Unit Tests<br/>Integration Tests<br/>Regression Tests] --> B
    K --> C
    K --> D
    K --> E
    K --> F
    K --> G
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style I fill:#fff3e0
    style J fill:#fff3e0
    style K fill:#fff3e0
```
