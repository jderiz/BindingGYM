## Data Processing

1.	Identification of Reference Sequences

To ensure the correct sequences are used, we manually identify the specific sequences of the two proteins expressed in the experiments by consulting the relevant literature. This step ensures that the sequences used in experiments are accurately matched to the reference sequences, which are often only a portion of the full UniProt gene sequences.

2. Search for Complex Crystal Structures in PDB

To ensure we use the most accurate structures available, we search for crystal structures of protein complexes in the Protein Data Bank (PDB). For publications with DMS data that lack complex structures, we seek the most recent and sequence-similar complex structures in the PDB to maximize the utility of these data. For example, at the time of the Heredia2018 publication, the crystal structure of the CXCR4_CXCL12 complex had not been published, and similarly, the CD19_FMC63 complex structure was unavailable when Klesmith2019 was published. The introduction of crystal structures not only aids in modeling but also enhances our understanding of protein-protein interactions.

3. Calculation of Binding Scores from Next-generation sequencing (NGS) Results

If a publication does not provide binding score results but includes raw sequencing data, we replicate the article’s data processing workflow to calculate the binding scores. The calculation methods, based on the DMS experimental design, are summarized as follows:

 1.	Calculate the log ratio of sequence frequencies between selected and unselected libraries, compared to the wild type (WT).
 2.	Calculate the log ratio of the frequency between the entire library before selection and the positive library after selection, then compare it to the WT ratio (fitness score or enrichment score).
 3.	For multi-partitioned screening based on different fluorescence intensities and concentrations, fit the Hill function to calculate -logKd.
 4.	For multi-round screening, fit a linear model to the corrected sequence frequencies; the slope indicates the degree of change.
 5.	Calculate changes in free energy.
 6.	For single-round screening, calculate the escape score (negated) for antibody escape from the virus. Some publications use a polyclonal approach to model escape scores across multiple concentrations.

The binding score reflects the effect of mutations on the affinity between two proteins. While the relative ranking of these scores correlates with the true change in binding affinity, they are not directly comparable across different groups due to variations in experimental conditions and calculation methods.

4.	Confirmation of Correct Mutation Sites

We confirm the accuracy of mutation sites by comparing them with the reference sequences.

5.	Consistency Verification

We compare the results with the main findings of the article to ensure consistency and reduce unnecessary errors due to human or accidental factors. For example, we identified a serialization error in the data from Heredia2018 and corrected it based on the original text.

6. Homology Modeling of Complex Structures

Since the complex crystal structures in the PDB often do not perfectly match the reference sequences provided in publications, we perform homology modeling of the complex structures using the reference sequences and existing crystal structures.

7. Dataset Refinement

For data from articles where confidence levels were measured, we filter the dataset using the original confidence levels provided. For raw data without confidence levels, we calculate the confidence for each sequence as the ratio of the sequence count to the total sequencing count (count/total). Sequences where log10(confidence) is less than a specified threshold, typically set at -5, are then removed. This threshold may be adjusted based on the data distribution of different assays to ensure that the proportion of removed data is not excessive.
