#!/bin/bash

mkdir datasets
cd datasets
mkdir Annots
wget https://worksheets.codalab.org/rest/bundles/0x36e16907b7254571b708b725f8beae52/contents/blob/ -O gwhd_2021.tar.gz
mkdir gwhd_2021
scp gwhd_2021.tar.gz ./gwhd_2021/
cd gwhd_2021
tar -zxvf gwhd_2021.tar.gz
scp ./official_train.csv ../Annots/
scp ./official_val.csv ../Annots/
scp ./official_test.csv ../Annots/
