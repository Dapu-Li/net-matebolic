---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Zhirong Li"
  text: "Yating Miao"
  tagline: Jilin University and China Medical University
  image:
    src:  /logo/IMG197.jpg
    alt: Logo
  actions:
    - theme: brand
      text: Disease
      link: /Disease/Baseline_Overview/
    - theme: brand
      text: Traits
      link: /Traits/Overview/
    - theme: brand
      text: Chapter
      link: /Chapter/Baseline_Overview
    - theme: brand
      text: Cluster
      link: /Cluster/Baseline_Overview
    - theme: alt
      text: Prediction and Diagnosis
      link: /Predict/Overview
    - theme: alt
      text: Mendelian Randomization
      link: /MR/Forward_Overview
    # - theme: alt
    #   text: Sensitivity Analysis
    #   link: /Sensitive/Baseline_Overview
    # - theme: alt
    #   text: Code for this project
    #   link: https://github.com/Dapu-Li/Plasma-metabolic


features:
  - title: Comorbidity and Disease Clustering
    details: >
      This study systematically evaluates the roles of plasma metabolites across 2,036 human diseases, with a focus on disease comorbidity. We identify key metabolites that drive co-occurring diseases. By clustering diseases based on shared metabolic associations, we propose a metabolite-driven reclassification of disease groups. This approach moves beyond traditional organ-based classification and provides novel insights into shared molecular pathways underlying comorbidity.

  - title: Predictive Modeling Across Disease Timelines
    details: >
      We developed and validated a series of predictive models based on metabolomics data, covering prevalent diseases, incident diseases within 5 years, within 10 years, and over the full follow-up period. These models systematically benchmark metabolite-based predictions against traditional demographic risk factors such as age, sex, and so on. Our findings demonstrate that metabolomics data offer robust, independent predictive power in short time horizons, enabling earlier disease detection and more precise risk stratification.

  - title: Code and Data Availability
    details: >
      All analysis code is publicly available on GitHub to ensure reproducibility and transparency. Metabolite and phenotype data were obtained from the UK Biobank and can be accessed through their standard data access process:  
      https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access.

