<h1 align="center">
  MADMESHR: Mixed-element ADvanced MESH generator with Reinforcement-learning for 2D hydrodynamic domains.
</h1>

<p align="center">
  <strong><a href="https://scholar.google.com/citations?user=IBFSkOcAAAAJ&hl=en">Dominik Mattioli</a><sup>1†</sup>, <a href="https://scholar.google.com/citations?user=mYPzjIwAAAAJ&hl=en">Ethan Kubatko</a><sup>2</sup></strong><br>
  <sup>†</sup>Corresponding author<br><br>
  <sup>1</sup>Penn State University<br>
  <sup>2</sup>Computational Hydrodynamics and Informatics Lab (CHIL), The Ohio State University
</p>

<p align="center">
  <a href="https://ceg.osu.edu/computational-hydrodynamics-and-informatics-laboratory">
    <img src="https://img.shields.io/badge/CHIL%20Lab%20@%20OSU-a7b1b7?logo=academia&logoColor=ba0c2f&labelColor=ba0c2f" alt="CHIL Lab @ OSU">
  </a>
  <a href="https://ceg.osu.edu/computational-hydrodynamics-and-informatics-laboratory">
    <img src="https://img.shields.io/badge/OSU_CHIL-ADMESH-66bb33?logo=github&logoColor=ba0c2f&labelColor=ffffff" alt="OSU CHIL ADMESH">
  </a>
  <a href="https://github.com/user-attachments/files/19724263/QuADMESH-Thesis.pdf">
    <img src="https://img.shields.io/badge/Thesis-QuADMESH-ba0c2f?style=flat-square&logo=book&logoColor=white&labelColor=cfd4d8" alt="QuADMESH Thesis">
  </a>
  <a href="https://github.com/domattioli/MADMESHR/blob/4cdc85418f2d357f28634365edde7a7f43ac99eb/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT">
  </a>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/265ae167-ad40-4bb3-82c6-1602d6d31ba7" alt="image", width="70%">
</p>

## Installation
```bash
git submodule add https://github.com/OSUdomattioli/CHILmesh.git external/CHILmesh
````

## Example Usage
```python
import sys
sys.path.append("external/CHILmesh")
from CHILmesh import CHILmesh  # or whatever the module exposes
```


Publications to inform codebase:
- [Triangular mesh generator with RL](https://arxiv.org/html/2504.03610v1)
- [FreeMeshRL](https://arxiv.org/abs/2203.11203)

