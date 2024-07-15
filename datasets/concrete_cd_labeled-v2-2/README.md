# About Dataset

The dataset contains 81 pairs of high-resolution RGB images of 2000x2000 size in png extension, where a pair depicts a vertically photographed circle of the same concrete and one of the images in the pair is irradiated and contains changes. The dataset contains binary masks for solving the problem of segmentation of changes caused by radiation.

All pairs were collected from 6 concrete samples named as 1A, 1C, 3A, 3C, 4A, 5A. Each sample contains from 11 to 14 pairs of images.

| Concrete Sample Name | Num. of Pairs |
| :------------------: | :-----------: |
|          1A          |      11      |
|          1C          |      14      |
|          3A          |      14      |
|          3C          |      14      |
|          4A          |      14      |
|          5A          |      14      |

The ``data`` directory contains pairs for each sample. Each pair contains two images named ``after_<№ of pair>.png`` and ``before_<№ of pair>.png``, '<№ of pair>' corresponds to the pair number inside one concrete sample, i.e. it is a number 1-11 or 1-14 depending on the number of pairs for the sample. The image with the prefix "before" corresponds to the image before irradiation, and the prefix "after" to the image after irradiation.

The ``masks`` directory contains masks for all 81 pairs and each mask has the format ``<sample name>_<pair>.PNG``. '`<sample name>`' is in [1A, 1C, 3A, 3C, 4A, 5A] and '<№ of pair>' is the pair number inside the sample `<sample name>`.

The masks were created manually and may contain inaccuracies.
