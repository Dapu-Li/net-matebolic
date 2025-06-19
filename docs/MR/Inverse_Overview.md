# Inverse Mendelian Randomization Analysis

## Objective

The purpose of the **Inverse Mendelian Randomization (MR)** analysis was to explore whether **diseases causally influence metabolite levels**, thereby evaluating potential **Inverse causality** in metabolite–disease relationships.

---

## Exposure Variables: Disease Outcomes

Genetic instruments for **disease exposures** were extracted from the **FinnGen R12** dataset. We included a total of **963 diseases**, consistent with those used in the forward MR analysis. These diseases were selected based on the following criteria:

1. Availability of high-quality genome-wide association study (GWAS) summary statistics.
2. Sufficient number of genome-wide significant SNPs (*P* < 5 × 10⁻⁸).
3. Biological relevance to metabolic alterations.

---

## Outcome Variables: Metabolites

The outcome variables were **251 metabolites and their ratios**, for which genetic effect sizes (beta coefficients and standard errors) were available from a large-scale GWAS in the UK Biobank, including:

- 249 metabolites from:  
  *Zhang S, et al. Nature Communications. 2024;15:8081.*
- 2 additional metabolites developed using the same approach:  
  - **Glucose-lactate (Glu-lactate)**  
  - **Spectrometer-corrected alanine (Sc-ala)**

---

## Instrument Selection Criteria

For each disease exposure, SNPs were selected as instruments based on the following:

1. **Significance**: SNPs associated with the disease at *P* < 5 × 10⁻⁸  
2. **Independence**: LD clumping was performed (r² < 0.01 within a 1 Mb window)  
3. **Exclusion of MHC region** (chr6: 25.5–34.0 Mb)  
4. **Instrument strength**: F-statistic > 10

---

## Analytical Methods

Inverse MR was performed using the **TwoSampleMR** package in **R**, following a strategy based on the number of instruments:

- `mr_wald_ratio`: For exposures with a single instrument  
- `mr_ivw`: For exposures with multiple independent instruments

This ensured appropriate modeling of causal estimates and minimized bias.

---

## Software and Tools

- Programming language: **R**
- MR package: [`TwoSampleMR`](https://mrcieu.github.io/TwoSampleMR/)
- Exposure data source: **FinnGen R12**
- Outcome data source: **UK Biobank Metabolomics GWAS**

---

## Multiple Testing Correction

Given the large number of tested disease–metabolite pairs (963 diseases × 251 metabolites), **Bonferroni correction** was applied to account for multiple comparisons.  
The significance threshold was set at:  
**p < 0.05 / (963 × 251)** ≈ **2.1 × 10⁻⁷**

---

## Interpretation

Significant results from the Inverse MR analysis may suggest that certain diseases contribute to **metabolic alterations**, highlighting possible **Inverse causality**, **disease progression markers**, or **metabolites as consequences rather than causes**.

## Code Availability

```r
rm(list = ls())
#setwd("~/Desktop/metabolic_all")
setwd("C:/Users/Administrator/Desktop/metabolic_all")

library(data.table)
library(TwoSampleMR)

snp_code <- fread("E:/MR/disease/finngen_R12_AB1_DERMATOPHYTOSIS.gz")[, c("#chrom", "pos", "alt", "ref","rsids")]
colnames(snp_code) <- c("chromosome", "base_pair_location", "effect_allele", "other_allele", "rsid")
setDT(snp_code)
snp_code[, `:=`(
  chromosome = as.character(chromosome),
  base_pair_location = as.integer(base_pair_location),
  effect_allele = as.character(effect_allele),
  other_allele = as.character(other_allele)
)]

setkeyv(snp_code, c("chromosome", "base_pair_location", "effect_allele", "other_allele"))
gwas_code <- read.csv('data/MR/gwas_code.csv')
gwas_code1 <- na.omit(read.csv('data/MR/gwas_code.csv'))
gwas_code2 <- gwas_code[is.na(gwas_code$"Study.Accession"), ]
gwas_code2[1,3] <- 'Glu'
gwas_code2[2,3] <- 'Scala'
colnames(gwas_code2)
IV_all <- as.data.frame(fread('E:/MR/IV/IVs_disease_total.csv'))

metabolism_gwase_path1 <- "E:/MR/meta"
metabolism_gwase_path2 <- "E:/MR/zm"
results_dir <- "Results/s7_MR_inverse"

metabolism_files1 <- list.files(metabolism_gwase_path1, pattern = "\\.tsv$", full.names = TRUE)
metabolism_files2 <- list.files(metabolism_gwase_path2, pattern = "\\.txt$", full.names = TRUE)

for (i in seq_along(metabolism_files1)) {
  metabolism_file <- metabolism_files1[i]
  metabolism_code <- gsub("\\.tsv$", "", basename(metabolism_file))
  metabolism_name <- gwas_code1[gwas_code1$Study.Accession == metabolism_code, "Abbreviations"]
  
  metabolism_name <- gsub("/", "_", metabolism_name)
  output_file <- paste0(results_dir, "/", metabolism_name, ".csv")
  if (file.exists(output_file)) {
    cat(sprintf("[%d/%d] Exist: %s\n", i, length(metabolism_files1), metabolism_name))
    next
  }
  
  cat(sprintf("[%d/%d] Ing: %s\n", i, length(metabolism_files1), metabolism_name))

  IV <- as.data.frame(IV_all[order(IV_all$P), ])
  ins <- format_data(
    IV, type = "exposure", header = TRUE, phenotype_col = "pheno",
    snp_col = "rsid", beta_col = "BETA", se_col = "SE", eaf_col = "A1_FREQ",
    effect_allele_col = "ALLELE1", other_allele_col = "ALLELE0", pval_col = "P"
  )
  
  tryCatch({
    
    m_data <- fread(metabolism_file, header = TRUE)
    m_data[, chromosome := as.character(chromosome)]
    setDT(m_data)
    setkeyv(m_data, c("chromosome", "base_pair_location", "effect_allele", "other_allele"))
    m_data <- merge(m_data, snp_code, by = c("chromosome", "base_pair_location", "effect_allele", "other_allele"), all.x = TRUE)
    m_data <- as.data.frame(m_data)
    
    m_data$Pro_code <- metabolism_name
    colnames(m_data)
    out <- format_data(
      m_data, type = "outcome", header = TRUE, snps = ins$SNP, phenotype_col = "Pro_code",
      snp_col = "rsid", beta_col = "beta", se_col = "standard_error",
      effect_allele_col = "effect_allele", other_allele_col = "other_allele",
      eaf_col = "effect_allele_frequency", pval_col = "p_value"
    )
    
    harmo <- harmonise_data(exposure_dat = ins, outcome_dat = out)
    
    mr_res <- mr(harmo, method_list = c("mr_wald_ratio", "mr_ivw"))
    results_OR <- generate_odds_ratios(mr_res)
    results_OR <- results_OR[, -c(1, 2)]
    
    
    fwrite(results_OR, output_file, col.names = TRUE, row.names = FALSE, quote = FALSE, sep = ",")
    
    cat(sprintf("Done: %s\n", metabolism_name))
    
  }, error = function(e) {
    cat(sprintf("Error: %s\n", metabolism_name))
    error_file <- paste0(results_dir, "/error_metabolism.txt")
    fwrite(data.table(metabolism_name), file = error_file, append = TRUE, row.names = FALSE, col.names = FALSE)
  })
}


for (i in seq_along(metabolism_files2)) {
  metabolism_file <- metabolism_files2[i]
  metabolism_code <- gsub("\\_file.txt$", "", basename(metabolism_file))
  metabolism_name <- gwas_code2[gwas_code2$"Study.Accession" == metabolism_code, "Abbreviations"]
  
  cat(sprintf("[%d/%d] Ing: %s\n", i, length(metabolism_files2), metabolism_name))
  
  IV <- as.data.frame(IV_all[order(IV_all$P), ])
  colnames(IV)
  ins <- format_data(
    IV, type = "exposure", header = TRUE, phenotype_col = "pheno",
    snp_col = "rsid", beta_col = "BETA", se_col = "SE", eaf_col = "A1_FREQ",
    effect_allele_col = "ALLELE1", other_allele_col = "ALLELE0", pval_col = "P"
  )
  
  tryCatch({
    
    m_data <- fread(metabolism_file, header = TRUE)
    colnames(m_data)
    m_data$Pro_code <- metabolism_name
    colnames(m_data)
    m_data <- as.data.frame(m_data)
    
    out <- format_data(
      m_data, type = "outcome", header = TRUE, snps = ins$SNP, phenotype_col = "Pro_code",
      snp_col = "ID", beta_col = "BETA", se_col = "SE",
      effect_allele_col = "A1", other_allele_col = "REF",
      eaf_col = "A1_FREQ", pval_col = "P"
    )
    
    harmo <- harmonise_data(exposure_dat = ins, outcome_dat = out)
    
    mr_res <- mr(harmo, method_list = c("mr_wald_ratio", "mr_ivw"))
    results_OR <- generate_odds_ratios(mr_res)
    results_OR <- results_OR[, -c(1, 2)]
    
    output_file <- paste0(results_dir, "/", metabolism_name, ".csv")
    fwrite(results_OR, output_file, col.names = TRUE, row.names = FALSE, quote = FALSE, sep = ",")
    
    cat(sprintf("Done: %s\n", metabolism_name))
    
  }, error = function(e) {
    cat(sprintf("Error: %s\n", metabolism_name))
    error_file <- paste0(results_dir, "/error_metabolism.txt")
    fwrite(data.table(metabolism_name), file = error_file, append = TRUE, row.names = FALSE, col.names = FALSE)
  })
}

cat("All is Done!\n")