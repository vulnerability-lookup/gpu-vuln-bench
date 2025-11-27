# Benchmarking GPU Performance in Vulnerability Severity Model Training


## Read the report

The report is available here:

- [gpu-performance-vuln-model.pdf](gpu-performance-vuln-model.pdf)
c

## Usage

### Update the charts

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python compare_experiments.py data/2_NVIDIA_L40S.csv data/2_NVIDIA_H100_NVL.csv data/4_NVIDIA_L40S.csv
```

The ``data`` folder should contain the raw CSV files of each experiment. Versioned on Hugging Face.


### Update the PDF report

```
pandoc experiments_report.md -f markdown-implicit_figures --toc -t pdf -o gpu-performance-vuln-model.pdf
```


## Funding

![EU Funding](europe.png)

[AIPITCH](https://www.science.nask.pl/en/research-areas/projects/12456) aims to create advanced artificial intelligence-based tools supporting key operational services in cyber defense.
These include technologies for early threat detection, automatic malware classification, and improvement of analytical processes through the integration of Large Language Models (LLM).
The project has the potential to set new standards in the cybersecurity industry.

The project leader is NASK National Research Institute. The international consortium includes:

- CIRCL (Computer Incident Response Center Luxembourg), Luxembourg
- The Shadowserver Foundation, Netherlands
- NCBJ (National Centre for Nuclear Research), Poland
- ABI LAB (Centre of Research and Innovation for Banks), Italy

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Cybersecurity Competence Centre.
Neither the European Union nor the European Cybersecurity Competence Centre can be held responsible for them.


## License

[This work](https://github.com/vulnerability-lookup/gpu-vuln-bench) is licensed under
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.html)

~~~
Copyright (c) 2025 Computer Incident Response Center Luxembourg (CIRCL)
Copyright (C) 2025 Cédric Bonhomme - https://github.com/cedricbonhomme
~~~

