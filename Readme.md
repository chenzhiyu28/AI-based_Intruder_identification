IMPARTANT: This project is the sole property of Zhiyu Chen and is submitted to the University of Queensland. All rights are reserved. No part of this project may be reproduced, distributed, or transmitted in any form or by any means without prior written permission from the owner.

[Zhiyu Chen / University of Queensland]
*************************************************************************************************************************************

![project poster](https://github.com/chenzhiyu28/AI-based_Intruder_identification/blob/master/1.jpg?raw=true)

This is a guidance of how the software works. Note some files may not be used at all. And some 
files are the output of the software.

Firstly, connect the hardware of AWR1843, then install the drivers, flash into the firmware, and connect the radar.

1.readData_AWR1843.py would start collecting data and store it with its timestamp as name (030318.py). (change the port if it is different on other computers)

all other remaining software are in the test folder.

2. plots.py allows to play back the recorded data. (remember to change the imported data file each time)

3. preprocessing.py contains several functions like smooth, imputation, outlier detection and also clustering from clustering.py. The input is the imported data file, and the output is digitized features of data file.

4. batch_process.py is for generating all features within dataset.

5. write_data.py is for put all the features of data into a csv file.

6. identification.py is for training the model with training data and generate predictions on testing data. Also the accuracy and f1 score is presented.
