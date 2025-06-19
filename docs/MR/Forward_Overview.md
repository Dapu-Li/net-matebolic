# Forward Mendelian Randomization Analysis

## Instrument Development

**Forward Mendelian Randomization (MR)** was performed to investigate the **causal effects of metabolites on disease outcomes**.

Among the **251 metabolites and their ratios**, genetic instruments for **249 metabolites** were obtained from the following publication:

> Zhang S, Wang Z, Wang Y, Zhu Y, Zhou Q, Jian X, et al.  
> *A metabolomic profile of biological aging in 250,341 individuals from the UK Biobank*.  
> Nature Communications. 2024; **15**:8081.

The remaining two metabolites were:

- **Glucose-lactate (Glu-lactate)**
- **Spectrometer-corrected alanine (Sc-ala)**

Genetic instruments for these two were developed using the **same methodology** as in the Zhang et al. study, ensuring consistency in instrument quality and selection criteria.

---

## Instrument Selection Criteria

To ensure the validity of the instruments, we applied the following stringent selection criteria:

1. **Genome-wide significance**: SNPs associated with metabolites at *P* < 5 × 10⁻⁸  
2. **Linkage disequilibrium (LD) clumping**:  
   - *r²* < 0.01  
   - Window size = 1 Mb  
3. **MHC region exclusion**:  
   - SNPs within the extended MHC region (chr6: 25.5–34.0 Mb) were excluded due to high LD and complexity  
4. **Instrument strength**:  
   - Only SNPs with an **F-statistic > 10** were retained to avoid weak instrument bias


---

## Disease Outcomes

The disease outcomes included in the MR analysis were obtained from the **FinnGen R12** dataset, which provides a comprehensive collection of genetically informed disease phenotypes.

We selected a total of **963 diseases**, representing the **union** of conditions relevant to our prior analyses.

A full list of the included diseases is available in the following section.

---

## Analytical Methods

MR analyses were conducted using the **TwoSampleMR** package in **R**. The selection of MR method was based on the **number of valid instruments per metabolite**:

- `mr_wald_ratio`:  
  Applied when **only one instrument** was available for a given exposure  
- `mr_ivw`:  
  Used when **multiple independent instruments** were available (inverse-variance weighted method)

This strategy ensures both **robustness** and **statistical power** across different metabolite-disease pairs.

---

## Software and Tools

- Language: **R**
- Package: [`TwoSampleMR`](https://mrcieu.github.io/TwoSampleMR/)

---

## Multiple Testing Correction

Given the large number of tested disease–metabolite pairs (963 diseases × 251 metabolites), **Bonferroni correction** was applied to account for multiple comparisons.  
The significance threshold was set at:  
**p < 0.05 / (963 × 251)** ≈ **2.1 × 10⁻⁷**

---


## Code for MR Analysis
```r
rm(list = ls())
#setwd("~/Desktop/metabolic_all")
setwd("C:/Users/Administrator/Desktop/metabolic_all")
library(data.table)
library(TwoSampleMR)

IV_all <- fread('E:/MR/IV/IVs_total.csv')

disease_gwase_path <- "E:/MR/disease"
results_dir <- "Results/s7_MR"

disease_files <- list.files(disease_gwase_path, pattern = "\\.gz$", full.names = TRUE)

for (i in seq_along(disease_files)) {
  disease_file <- disease_files[i]
  disease_name <- gsub("\\.gz$", "", basename(disease_file))
  disease_name <- gsub("finngen_R12_", "", disease_name)
  
  output_file <- paste0(results_dir, "/", disease_name, ".csv")
  if (file.exists(output_file)) {
    cat(sprintf("[%d/%d] Exist: %s\n", i, length(disease_files), disease_name))
    next
  }
  
  cat(sprintf("[%d/%d] Ing: %s\n", i, length(disease_files), disease_name))
  
  disease_data <- as.data.frame(fread(disease_file, header = TRUE))
  
  IV <- as.data.frame(IV_all[order(IV_all$P), ])
  ins <- format_data(
    IV, type = "exposure", header = TRUE, phenotype_col = "Pro_code",
    snp_col = "rsid", beta_col = "BETA", se_col = "SE", eaf_col = "A1_FREQ",
    effect_allele_col = "ALLELE1", other_allele_col = "ALLELE0", pval_col = "P"
  )
  
  tryCatch({
    disease_data$pheno <- disease_name
    out <- format_data(
      disease_data, type = "outcome", header = TRUE, snps = ins$SNP, phenotype_col = "pheno",
      snp_col = "rsids", beta_col = "beta", se_col = "sebeta",
      effect_allele_col = "alt", other_allele_col = "ref",
      eaf_col = "af_alt", pval_col = "pval"
    )

    harmo <- harmonise_data(exposure_dat = ins, outcome_dat = out)
    
    mr_res <- mr(harmo, method_list = c("mr_wald_ratio", "mr_ivw"))
    results_OR <- generate_odds_ratios(mr_res)
    results_OR <- results_OR[, -c(1, 2)]
    
    fwrite(results_OR, output_file, col.names = TRUE, row.names = FALSE, quote = FALSE, sep = ",")
    
    cat(sprintf("Done: %s\n", disease_name))
    
  }, error = function(e) {
    cat(sprintf("Error: %s\n", disease_name))
    error_file <- paste0(results_dir, "/error_diseases.txt")
    fwrite(data.table(disease_name), file = error_file, append = TRUE, row.names = FALSE, col.names = FALSE)
  })
}

cat("All is Done!\n")
```