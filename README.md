# T2MAT(Text-to-Material)


  A comprehensive framework processing from a **user-input sentence**, generating material structures aligning with goal properties beyond the existing database via globally exploring chemical space, followed by an entirely automated Density Functional Theory (DFT) validation workflow.
  
  
![图片](https://github.com/szl666/T2MAT/assets/44625390/d5949dce-34d3-481f-9f39-eac0ce534625)


In the current version of T2MAT, we developed a modified CDVAE with improved symmetry for generating material structures. The CGTNet is a new GNN with improved accuracy and small data friendliness compared to CGCNN, SchNet, DimeNet++, PaiNN and GemNet. The complete version of T2MAT including **local LLM, CGTNet, modified CDVAE and automatic DFT framework** will be updated after the manuscript of T2MAT is accepted.


Furthermore, we have developed a **new generative model——SymGEN**. The **symmetry**(ratio of structures with non-P1 space group, **Figure e**), **thermodynamic stability**(DFT-calculated formation energies, **Figure f**) and **kinetic stability**(DFT-calculated phonon properties, **Figure g**) of generated structures are significantly improved compared to the CDVAE. 

SymGEN will be replaced with the CDVAE in T2MAT after the manuscript of SymGEN is accepted.


![image](https://github.com/szl666/inverse_design/assets/44625390/0e5c0aff-6840-4cd6-a5a9-df32d1613f3b)
