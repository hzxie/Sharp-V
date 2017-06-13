# Similar Patient Finder

## Introduction

The Similar Patient Finder (SPF), a web-based molecular level aided diagnosis tool, is aimed at helping physicians explore the patients’ similarity at the molecular level and facilitate prescription making. The SPF integrates flexible medical data, including molecular level high-throughput data and corresponding clinical data if available. Some methods are provided to easily handle the related data and visualize it in different forms for auxiliary diagnosis.

## Installation

Make sure you have Python 2, MySQL, pip installed.

To be started, you have to install Python dependencies:

```
pip install -r requirements.txt
```

Then you need to edit configuration of the application in `server.conf`.

After that, use following command to start web server:

```
python Sharp-V.py
```

## License

This project is open sourced under [GNU GPL v2](http://www.gnu.org/licenses/gpl-2.0.txt).

