# SklLib

A toolbox to test deeplearning models on Wi-Fi Human Recognition.

## Installation

Try to execute below script on a linux computer with conda and git:

```bash
git clone [url]
cd [wilib]
conda create -n wifilib python==3.7
conda activate wifilib
pip install -r requirements.txt
```

## Run

```bash
python ./run.py -config [path to config] -debug False
```

The example configurations are under the 'config' directory.

## Model Zoo

<table>
    <thead>
    <tr>
        <th>Model</th>
        <th>Paper</th>
        <th>Code</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>ST-GCN (AAAI 2018)</td>
        <td><a href="https://arxiv.org/abs/1801.07455">
            Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition</a></td>
        <td><a href="https://github.com/kennymckormick/pyskl">https://github.com/kennymckormick/pyskl</a></td>
    </tr>
    </tbody>
</table>

## Datasets

<table>
    <thead>
    <tr>
        <th>Dataset Name</th>
        <th>Paper</th>
        <th>Data resource</th>
        <th>Benchmark</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>NTU RGB+D</td>
        <td><a href=""></a></td>
        <td><a href=""></a></td>
         <td><a href=""></a></td>
    </tr>
    </tbody>
</table>
