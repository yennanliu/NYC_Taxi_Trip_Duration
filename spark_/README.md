# Run the model again with Spark Mlib via Java/Scala/Python 



## Demo 

```bash 

# run Spark RF model with Java
# compile 
$ javac -classpath  ".:/Users/$USER/spark/jars/*" train_spark_RF.java
# run the java spark ML model 
$ java -classpath  ".:/Users/$USER/spark/jars/*"  train_spark_RF





# ------------------ output  ------------------
....

2018-08-03 08:21:11 INFO  DAGScheduler:54 - ResultStage 1 (show at train_spark_RF.java:76) finished in 0.475 s
2018-08-03 08:21:11 INFO  DAGScheduler:54 - Job 1 finished: show at train_spark_RF.java:76, took 0.482521 s
+---------+---------+---------------+------------------+------------------+------------------+------------------+-------------+--------------------+
|       id|vendor_id|passenger_count|  pickup_longitude|   pickup_latitude| dropoff_longitude|  dropoff_latitude|trip_duration|            features|
+---------+---------+---------------+------------------+------------------+------------------+------------------+-------------+--------------------+
|id0000967|      2.0|            1.0|-73.97216033935547| 40.75592422485352|-73.98228454589844| 40.75704193115234|        299.0|[2.0,1.0,-73.9721...|
|id0002063|      2.0|            1.0| -73.9947738647461| 40.75497055053711|-73.99646759033203|40.745277404785156|        496.0|[2.0,1.0,-73.9947...|
|id0010677|      1.0|            2.0|-73.99197387695312|40.749996185302734| -73.9778823852539|  40.7548942565918|        769.0|[1.0,2.0,-73.9919...|
|id0012891|      2.0|            1.0|-73.98104858398438| 40.74433898925781| -73.9729995727539| 40.78998947143555|       1225.0|[2.0,1.0,-73.9810...|

....
  total: 2.668285723
  findSplits: 0.890998687
  findBestSplits: 1.337335943
  chooseSplits: 1.325597905

....

+---------+-------------+------------------+
|       id|trip_duration|        prediction|
+---------+-------------+------------------+
|id0000569|        123.0| 696.5536191440954|
|id0005214|        751.0| 739.6998510527386|
|id0015493|       2451.0|1672.4643407219774|
|id0021343|       1074.0| 640.1038746046298|
|id0088231|       1030.0| 766.7510769207481|
|id0153673|        604.0| 707.4976739458716|
|id0204192|        896.0| 1372.572467343693|
|id0214120|        655.0| 734.4709601056011|
|id0217206|        543.0| 753.7172912732649|
|id0222570|        926.0| 709.9500717425739|
|id0244490|       1319.0| 816.5032144254858|
|id0244668|        737.0| 728.3853269760641|
|id0260905|        711.0| 773.8947306866996|
|id0265651|        224.0| 729.8960894921888|
|id0290690|        218.0| 750.2640061306786|
|id0301361|        459.0|  677.500862852768|
|id0309774|        199.0| 729.5973998432885|
|id0317057|       1370.0| 1405.089356062103|
|id0350661|       1912.0| 853.2485872298719|
|id0381402|        771.0| 830.3946300617876|
+---------+-------------+------------------+
only showing top 20 rows

----------------- prediction ----------------- 
----------------- accuracy ----------------- 
Trained RF model:
RandomForestRegressionModel (uid=rfr_2018fa70139b) with 20 trees
  Tree 0 (weight 1.0):
    If (feature 2 <= -73.93980407714844)
     If (feature 5 <= 40.723501205444336)
      If (feature 2 <= -73.99408340454102)
       If (feature 4 <= -73.96314239501953)
        If (feature 5 <= 40.70528793334961)
         Predict: 1041.4
        Else (feature 5 > 40.70528793334961)
         Predict: 670.0714285714286
       Else (feature 4 > -73.96314239501953)
        If (feature 0 <= 1.5)
         Predict: 1069.0
        Else (feature 0 > 1.5)
         Predict: 1524.0
      Else (feature 2 > -73.99408340454102)
       If (feature 2 <= -73.98944091796875)
        If (feature 0 <= 1.5)
         Predict: 2019.6363636363637
        Else (feature 0 > 1.5)
         Predict: 2203.6666666666665
       Else (feature 2 > -73.98944091796875)
        If (feature 4 <= -73.94768524169922)
         Predict: 1141.0
        Else (feature 4 > -73.94768524169922)
         Predict: 1847.5555555555557
     Else (feature 5 > 40.723501205444336)
      If (feature 4 <= -73.93783187866211)
       If (feature 5 <= 40.78069496154785)
        If (feature 3 <= 40.73097038269043)
         Predict: 1019.275
        Else (feature 3 > 40.73097038269043)
         Predict: 689.5224586288416
       Else (feature 5 > 40.78069496154785)
        If (feature 1 <= 5.5)
         Predict: 558.0
        Else (feature 1 > 5.5)
         Predict: 174.0
      Else (feature 4 > -73.93783187866211)
       If (feature 1 <= 2.5)
        If (feature 3 <= 40.743438720703125)
         Predict: 1203.3333333333333
        Else (feature 3 > 40.743438720703125)
         Predict: 1660.6923076923076
       Else (feature 1 > 2.5)
        Predict: 2534.0
    Else (feature 2 > -73.93980407714844)
     If (feature 5 <= 40.73819923400879)
      If (feature 3 <= 40.6946907043457)
       If (feature 1 <= 3.5)
        If (feature 0 <= 1.5)
         Predict: 1934.75
        Else (feature 0 > 1.5)
         Predict: 1784.0
       Else (feature 1 > 3.5)
        Predict: 2453.0
      Else (feature 3 > 40.6946907043457)
       If (feature 4 <= -73.98495483398438)
        Predict: 706.0
       Else (feature 4 > -73.98495483398438)
        If (feature 2 <= -73.8634262084961)
         Predict: 878.6666666666666
        Else (feature 2 > -73.8634262084961)
         Predict: 814.0
     Else (feature 5 > 40.73819923400879)
      If (feature 4 <= -73.98495483398438)
       If (feature 0 <= 1.5)
        If (feature 1 <= 1.5)
         Predict: 1305.0
        Else (feature 1 > 1.5)
         Predict: 2817.0
       Else (feature 0 > 1.5)
        If (feature 2 <= -73.8634262084961)
         Predict: 2967.0
        Else (feature 2 > -73.8634262084961)
         Predict: 6187.0
      Else (feature 4 > -73.98495483398438)
       If (feature 5 <= 40.79084587097168)
        If (feature 3 <= 40.6946907043457)
         Predict: 2424.5555555555557
        Else (feature 3 > 40.6946907043457)
         Predict: 1770.5
       Else (feature 5 > 40.79084587097168)
        If (feature 1 <= 1.5)
         Predict: 1524.5
        Else (feature 1 > 1.5)
         Predict: 1654.3333333333333
  Tree 1 (weight 1.0):
    If (feature 5 <= 40.690895080566406)
     If (feature 1 <= 2.5)
      If (feature 0 <= 1.5)
       If (feature 1 <= 1.5)
        Predict: 1158.0833333333333
       Else (feature 1 > 1.5)
        If (feature 2 <= -73.97146987915039)
         Predict: 2543.0
        Else (feature 2 > -73.97146987915039)
         Predict: 1733.0
      Else (feature 0 > 1.5)
       If (feature 2 <= -73.96899795532227)
        If (feature 1 <= 1.5)
         Predict: 1232.857142857143
        Else (feature 1 > 1.5)
         Predict: 1227.6
       Else (feature 2 > -73.96899795532227)
        Predict: 2120.0
     Else (feature 1 > 2.5)
      Predict: 3528.0
    Else (feature 5 > 40.690895080566406)
     If (feature 2 <= -73.93980407714844)
      If (feature 4 <= -73.93783187866211)
       If (feature 2 <= -74.00891876220703)
        If (feature 0 <= 1.5)
         Predict: 787.0
        Else (feature 0 > 1.5)
         Predict: 1396.4545454545455
       Else (feature 2 > -74.00891876220703)
        If (feature 5 <= 40.71343803405762)
         Predict: 916.1538461538462
        Else (feature 5 > 40.71343803405762)
         Predict: 701.3896103896104
      Else (feature 4 > -73.93783187866211)
       If (feature 3 <= 40.743438720703125)
        If (feature 1 <= 1.5)
         Predict: 1222.2222222222222
        Else (feature 1 > 1.5)
         Predict: 1466.0
       Else (feature 3 > 40.743438720703125)
        If (feature 1 <= 2.5)
         Predict: 1787.4545454545455
        Else (feature 1 > 2.5)
         Predict: 2534.0
     Else (feature 2 > -73.93980407714844)
      If (feature 2 <= -73.8634262084961)
       If (feature 3 <= 40.772178649902344)
        If (feature 3 <= 40.740970611572266)
         Predict: 604.0
        Else (feature 3 > 40.740970611572266)
         Predict: 1078.5
       Else (feature 3 > 40.772178649902344)
        If (feature 3 <= 40.77609443664551)
         Predict: 1777.0
        Else (feature 3 > 40.77609443664551)
         Predict: 893.5
      Else (feature 2 > -73.8634262084961)
       If (feature 4 <= -73.97111511230469)
        If (feature 5 <= 40.723501205444336)
         Predict: 1782.0
        Else (feature 5 > 40.723501205444336)
         Predict: 2628.3333333333335
       Else (feature 4 > -73.97111511230469)
        If (feature 5 <= 40.71343803405762)
         Predict: 1784.0
        Else (feature 5 > 40.71343803405762)
         Predict: 894.25
  Tree 2 (weight 1.0):
    If (feature 0 <= 1.5)
     If (feature 3 <= 40.6946907043457)
      If (feature 1 <= 1.5)
       If (feature 5 <= 40.690895080566406)
        If (feature 4 <= -73.96831512451172)
         Predict: 2824.0
        Else (feature 4 > -73.96831512451172)
         Predict: 580.0
       Else (feature 5 > 40.690895080566406)
        Predict: 2264.6666666666665
      Else (feature 1 > 1.5)
       If (feature 2 <= -73.97146987915039)
        Predict: 668.0
       Else (feature 2 > -73.97146987915039)
        If (feature 1 <= 2.5)
         Predict: 1733.0
        Else (feature 1 > 2.5)
         Predict: 2065.0
     Else (feature 3 > 40.6946907043457)
      If (feature 3 <= 40.78046417236328)
       If (feature 3 <= 40.768455505371094)
        If (feature 1 <= 3.5)
         Predict: 868.1813953488372
        Else (feature 1 > 3.5)
         Predict: 788.0
       Else (feature 3 > 40.768455505371094)
        If (feature 2 <= -73.93980407714844)
         Predict: 881.7692307692307
        Else (feature 2 > -73.93980407714844)
         Predict: 2034.6923076923076
      Else (feature 3 > 40.78046417236328)
       If (feature 5 <= 40.767757415771484)
        If (feature 3 <= 40.79401969909668)
         Predict: 2337.0
        Else (feature 3 > 40.79401969909668)
         Predict: 1152.75
       Else (feature 5 > 40.767757415771484)
        If (feature 4 <= -73.93783187866211)
         Predict: 423.35483870967744
        Else (feature 4 > -73.93783187866211)
         Predict: 1095.0
    Else (feature 0 > 1.5)
     If (feature 4 <= -73.93783187866211)
      If (feature 1 <= 3.5)
       If (feature 2 <= -73.8634262084961)
        If (feature 3 <= 40.734697341918945)
         Predict: 943.8695652173913
        Else (feature 3 > 40.734697341918945)
         Predict: 664.8949416342413
       Else (feature 2 > -73.8634262084961)
        Predict: 2635.3333333333335
      Else (feature 1 > 3.5)
       If (feature 4 <= -73.95264053344727)
        If (feature 2 <= -73.93980407714844)
         Predict: 641.8205128205128
        Else (feature 2 > -73.93980407714844)
         Predict: 2388.4
       Else (feature 4 > -73.95264053344727)
        If (feature 2 <= -74.00333023071289)
         Predict: 1280.0
        Else (feature 2 > -74.00333023071289)
         Predict: 1537.0
     Else (feature 4 > -73.93783187866211)
      If (feature 4 <= -73.87130355834961)
       If (feature 1 <= 2.5)
        If (feature 5 <= 40.77510643005371)
         Predict: 1829.75
        Else (feature 5 > 40.77510643005371)
         Predict: 1146.857142857143
       Else (feature 1 > 2.5)
        Predict: 598.0
      Else (feature 4 > -73.87130355834961)
       If (feature 3 <= 40.6946907043457)
        Predict: 3528.0
       Else (feature 3 > 40.6946907043457)
        If (feature 1 <= 1.5)
         Predict: 1663.0
        Else (feature 1 > 1.5)
         Predict: 1565.5
  Tree 3 (weight 1.0):
    If (feature 2 <= -73.8634262084961)
     If (feature 4 <= -73.93783187866211)
      If (feature 5 <= 40.72003936767578)
       If (feature 3 <= 40.747825622558594)
        If (feature 5 <= 40.690895080566406)
         Predict: 1056.5454545454545
        Else (feature 5 > 40.690895080566406)
         Predict: 671.3214285714286
       Else (feature 3 > 40.747825622558594)
        If (feature 0 <= 1.5)
         Predict: 2148.5714285714284
        Else (feature 0 > 1.5)
         Predict: 1480.0
      Else (feature 5 > 40.72003936767578)
       If (feature 4 <= -73.9727897644043)
        If (feature 3 <= 40.71845626831055)
         Predict: 1381.5714285714287
        Else (feature 3 > 40.71845626831055)
         Predict: 732.031545741325
       Else (feature 4 > -73.9727897644043)
        If (feature 2 <= -73.93980407714844)
         Predict: 591.0885416666666
        Else (feature 2 > -73.93980407714844)
         Predict: 1784.25
     Else (feature 4 > -73.93783187866211)
      If (feature 5 <= 40.77285957336426)
       If (feature 1 <= 2.5)
        If (feature 2 <= -73.95446014404297)
         Predict: 1532.5294117647059
        Else (feature 2 > -73.95446014404297)
         Predict: 565.5
       Else (feature 1 > 2.5)
        Predict: 598.0
      Else (feature 5 > 40.77285957336426)
       If (feature 2 <= -73.96899795532227)
        If (feature 0 <= 1.5)
         Predict: 2299.2
        Else (feature 0 > 1.5)
         Predict: 1466.0
       Else (feature 2 > -73.96899795532227)
        If (feature 0 <= 1.5)
         Predict: 1694.0
        Else (feature 0 > 1.5)
         Predict: 917.5
    Else (feature 2 > -73.8634262084961)
     If (feature 0 <= 1.5)
      Predict: 2387.1111111111113
     Else (feature 0 > 1.5)
      If (feature 4 <= -73.98495483398438)
       Predict: 6187.0
      Else (feature 4 > -73.98495483398438)
       If (feature 4 <= -73.9727897644043)
        If (feature 4 <= -73.9747200012207)
         Predict: 2699.0
        Else (feature 4 > -73.9747200012207)
         Predict: 3370.0
       Else (feature 4 > -73.9727897644043)
        Predict: 1924.4285714285713
  Tree 4 (weight 1.0):
    If (feature 5 <= 40.71343803405762)
     If (feature 3 <= 40.74550437927246)
      If (feature 1 <= 3.5)
       If (feature 5 <= 40.70528793334961)
        If (feature 2 <= -73.9784049987793)
         Predict: 1101.2307692307693
        Else (feature 2 > -73.9784049987793)
         Predict: 591.3
       Else (feature 5 > 40.70528793334961)
        If (feature 1 <= 1.5)
         Predict: 762.5
        Else (feature 1 > 1.5)
         Predict: 679.0
      Else (feature 1 > 3.5)
       Predict: 350.0
     Else (feature 3 > 40.74550437927246)
      If (feature 1 <= 2.5)
       If (feature 4 <= -73.98495483398438)
        If (feature 5 <= 40.70528793334961)
         Predict: 1385.5714285714287
        Else (feature 5 > 40.70528793334961)
         Predict: 1725.0
       Else (feature 4 > -73.98495483398438)
        If (feature 5 <= 40.70528793334961)
         Predict: 2102.8571428571427
        Else (feature 5 > 40.70528793334961)
         Predict: 2456.0
      Else (feature 1 > 2.5)
       Predict: 1093.0
    Else (feature 5 > 40.71343803405762)
     If (feature 0 <= 1.5)
      If (feature 3 <= 40.6946907043457)
       If (feature 5 <= 40.73819923400879)
        Predict: 1777.0
       Else (feature 5 > 40.73819923400879)
        Predict: 2566.0
      Else (feature 3 > 40.6946907043457)
       If (feature 2 <= -73.93980407714844)
        If (feature 4 <= -73.93783187866211)
         Predict: 725.3265306122449
        Else (feature 4 > -73.93783187866211)
         Predict: 1808.1333333333334
       Else (feature 2 > -73.93980407714844)
        If (feature 1 <= 1.5)
         Predict: 1079.3333333333333
        Else (feature 1 > 1.5)
         Predict: 2070.8333333333335
     Else (feature 0 > 1.5)
      If (feature 2 <= -73.8634262084961)
       If (feature 2 <= -73.93980407714844)
        If (feature 4 <= -73.93783187866211)
         Predict: 651.8615384615384
        Else (feature 4 > -73.93783187866211)
         Predict: 1585.6
       Else (feature 2 > -73.93980407714844)
        If (feature 3 <= 40.772178649902344)
         Predict: 713.3333333333334
        Else (feature 3 > 40.772178649902344)
         Predict: 1710.875
      Else (feature 2 > -73.8634262084961)
       If (feature 4 <= -73.98495483398438)
        If (feature 1 <= 1.5)
         Predict: 6187.0
        Else (feature 1 > 1.5)
         Predict: 2453.0
       Else (feature 4 > -73.98495483398438)
        Predict: 2699.0
  Tree 5 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 2 <= -73.96899795532227)
      If (feature 0 <= 1.5)
       If (feature 2 <= -73.9864501953125)
        Predict: 835.0
       Else (feature 2 > -73.9864501953125)
        Predict: 668.0
      Else (feature 0 > 1.5)
       If (feature 5 <= 40.70528793334961)
        If (feature 5 <= 40.690895080566406)
         Predict: 505.0
        Else (feature 5 > 40.690895080566406)
         Predict: 568.0
       Else (feature 5 > 40.70528793334961)
        Predict: 853.0
     Else (feature 2 > -73.96899795532227)
      If (feature 0 <= 1.5)
       Predict: 2016.5714285714287
      Else (feature 0 > 1.5)
       If (feature 5 <= 40.75032424926758)
        If (feature 4 <= -73.97111511230469)
         Predict: 2264.5
        Else (feature 4 > -73.97111511230469)
         Predict: 3528.0
       Else (feature 5 > 40.75032424926758)
        Predict: 5248.0
    Else (feature 3 > 40.6946907043457)
     If (feature 1 <= 4.5)
      If (feature 5 <= 40.690895080566406)
       If (feature 4 <= -73.87130355834961)
        If (feature 3 <= 40.72142791748047)
         Predict: 634.5
        Else (feature 3 > 40.72142791748047)
         Predict: 1534.25
       Else (feature 4 > -73.87130355834961)
        If (feature 0 <= 1.5)
         Predict: 2743.0
        Else (feature 0 > 1.5)
         Predict: 2485.0
      Else (feature 5 > 40.690895080566406)
       If (feature 5 <= 40.77736473083496)
        If (feature 2 <= -73.95005798339844)
         Predict: 820.9698189134809
        Else (feature 2 > -73.95005798339844)
         Predict: 1320.0645161290322
       Else (feature 5 > 40.77736473083496)
        If (feature 3 <= 40.75063514709473)
         Predict: 1141.6
        Else (feature 3 > 40.75063514709473)
         Predict: 593.223880597015
     Else (feature 1 > 4.5)
      If (feature 2 <= -73.97331619262694)
       If (feature 4 <= -73.9747200012207)
        If (feature 1 <= 5.5)
         Predict: 571.2857142857143
        Else (feature 1 > 5.5)
         Predict: 458.7307692307692
       Else (feature 4 > -73.9747200012207)
        If (feature 1 <= 5.5)
         Predict: 965.0
        Else (feature 1 > 5.5)
         Predict: 780.5
      Else (feature 2 > -73.97331619262694)
       If (feature 1 <= 5.5)
        If (feature 3 <= 40.76089668273926)
         Predict: 927.1428571428571
        Else (feature 3 > 40.76089668273926)
         Predict: 393.6
       Else (feature 1 > 5.5)
        If (feature 2 <= -73.96648406982422)
         Predict: 1499.0
        Else (feature 2 > -73.96648406982422)
         Predict: 951.0
  Tree 6 (weight 1.0):
    If (feature 2 <= -73.93980407714844)
     If (feature 5 <= 40.71343803405762)
      If (feature 3 <= 40.74550437927246)
       If (feature 1 <= 1.5)
        If (feature 3 <= 40.72142791748047)
         Predict: 692.6
        Else (feature 3 > 40.72142791748047)
         Predict: 1128.2857142857142
       Else (feature 1 > 1.5)
        If (feature 5 <= 40.690895080566406)
         Predict: 890.6666666666666
        Else (feature 5 > 40.690895080566406)
         Predict: 575.75
      Else (feature 3 > 40.74550437927246)
       If (feature 5 <= 40.690895080566406)
        If (feature 4 <= -73.98495483398438)
         Predict: 1393.0
        Else (feature 4 > -73.98495483398438)
         Predict: 2285.285714285714
       Else (feature 5 > 40.690895080566406)
        If (feature 1 <= 2.5)
         Predict: 1801.8333333333333
        Else (feature 1 > 2.5)
         Predict: 1022.0
     Else (feature 5 > 40.71343803405762)
      If (feature 3 <= 40.71845626831055)
       If (feature 5 <= 40.74869728088379)
        If (feature 2 <= -74.00597763061523)
         Predict: 600.2
        Else (feature 2 > -74.00597763061523)
         Predict: 888.125
       Else (feature 5 > 40.74869728088379)
        If (feature 2 <= -74.00597763061523)
         Predict: 1888.3
        Else (feature 2 > -74.00597763061523)
         Predict: 1169.0
      Else (feature 3 > 40.71845626831055)
       If (feature 2 <= -73.95005798339844)
        If (feature 0 <= 1.5)
         Predict: 750.0894941634241
        Else (feature 0 > 1.5)
         Predict: 688.0268456375838
       Else (feature 2 > -73.95005798339844)
        If (feature 3 <= 40.78476333618164)
         Predict: 366.59090909090907
        Else (feature 3 > 40.78476333618164)
         Predict: 553.0
    Else (feature 2 > -73.93980407714844)
     If (feature 4 <= -73.96831512451172)
      If (feature 5 <= 40.767757415771484)
       If (feature 2 <= -73.8634262084961)
        If (feature 3 <= 40.77383041381836)
         Predict: 2022.1666666666667
        Else (feature 3 > 40.77383041381836)
         Predict: 876.0
       Else (feature 2 > -73.8634262084961)
        If (feature 3 <= 40.6946907043457)
         Predict: 2232.4
        Else (feature 3 > 40.6946907043457)
         Predict: 2817.0
      Else (feature 5 > 40.767757415771484)
       If (feature 5 <= 40.77285957336426)
        Predict: 2566.0
       Else (feature 5 > 40.77285957336426)
        Predict: 2629.0
     Else (feature 4 > -73.96831512451172)
      If (feature 5 <= 40.71343803405762)
       If (feature 1 <= 1.5)
        If (feature 0 <= 1.5)
         Predict: 580.0
        Else (feature 0 > 1.5)
         Predict: 1784.0
       Else (feature 1 > 1.5)
        Predict: 3528.0
      Else (feature 5 > 40.71343803405762)
       If (feature 5 <= 40.74869728088379)
        If (feature 5 <= 40.73819923400879)
         Predict: 998.0
        Else (feature 5 > 40.73819923400879)
         Predict: 470.6666666666667
       Else (feature 5 > 40.74869728088379)
        If (feature 4 <= -73.93783187866211)
         Predict: 1439.5
        Else (feature 4 > -73.93783187866211)
         Predict: 843.0
  Tree 7 (weight 1.0):
    If (feature 2 <= -73.8634262084961)
     If (feature 3 <= 40.77609443664551)
      If (feature 3 <= 40.76089668273926)
       If (feature 4 <= -73.9727897644043)
        If (feature 2 <= -74.00891876220703)
         Predict: 961.1923076923077
        Else (feature 2 > -74.00891876220703)
         Predict: 682.3260073260074
       Else (feature 4 > -73.9727897644043)
        If (feature 5 <= 40.70528793334961)
         Predict: 1575.375
        Else (feature 5 > 40.70528793334961)
         Predict: 957.1931818181819
      Else (feature 3 > 40.76089668273926)
       If (feature 5 <= 40.71343803405762)
        If (feature 3 <= 40.762447357177734)
         Predict: 1327.0
        Else (feature 3 > 40.762447357177734)
         Predict: 2203.777777777778
       Else (feature 5 > 40.71343803405762)
        If (feature 5 <= 40.752187728881836)
         Predict: 1201.8823529411766
        Else (feature 5 > 40.752187728881836)
         Predict: 729.5045871559633
     Else (feature 3 > 40.77609443664551)
      If (feature 5 <= 40.77005386352539)
       If (feature 0 <= 1.5)
        If (feature 5 <= 40.74338722229004)
         Predict: 675.25
        Else (feature 5 > 40.74338722229004)
         Predict: 1058.9285714285713
       Else (feature 0 > 1.5)
        If (feature 5 <= 40.74611854553223)
         Predict: 843.0
        Else (feature 5 > 40.74611854553223)
         Predict: 1352.5
      Else (feature 5 > 40.77005386352539)
       If (feature 1 <= 2.5)
        If (feature 4 <= -73.98209762573242)
         Predict: 816.0
        Else (feature 4 > -73.98209762573242)
         Predict: 447.7857142857143
       Else (feature 1 > 2.5)
        If (feature 2 <= -73.9809341430664)
         Predict: 289.0
        Else (feature 2 > -73.9809341430664)
         Predict: 174.0
    Else (feature 2 > -73.8634262084961)
     If (feature 0 <= 1.5)
      If (feature 4 <= -73.96831512451172)
       If (feature 1 <= 1.5)
        If (feature 5 <= 40.73424530029297)
         Predict: 2469.0
        Else (feature 5 > 40.73424530029297)
         Predict: 3837.5
       Else (feature 1 > 1.5)
        If (feature 4 <= -74.00594329833984)
         Predict: 1733.0
        Else (feature 4 > -74.00594329833984)
         Predict: 2065.0
      Else (feature 4 > -73.96831512451172)
       Predict: 693.6666666666666
     Else (feature 0 > 1.5)
      If (feature 4 <= -73.98495483398438)
       Predict: 6187.0
      Else (feature 4 > -73.98495483398438)
       If (feature 4 <= -73.9747200012207)
        If (feature 4 <= -73.97915649414062)
         Predict: 2268.0
        Else (feature 4 > -73.97915649414062)
         Predict: 2699.0
       Else (feature 4 > -73.9747200012207)
        Predict: 1937.5
  Tree 8 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 5 <= 40.73424530029297)
      If (feature 4 <= -73.98324966430664)
       If (feature 2 <= -73.96899795532227)
        If (feature 2 <= -73.9958724975586)
         Predict: 853.0
        Else (feature 2 > -73.9958724975586)
         Predict: 526.0
       Else (feature 2 > -73.96899795532227)
        If (feature 4 <= -74.00594329833984)
         Predict: 1733.0
        Else (feature 4 > -74.00594329833984)
         Predict: 2114.0
      Else (feature 4 > -73.98324966430664)
       If (feature 0 <= 1.5)
        Predict: 1702.0
       Else (feature 0 > 1.5)
        Predict: 3528.0
     Else (feature 5 > 40.73424530029297)
      If (feature 5 <= 40.77285957336426)
       If (feature 5 <= 40.752187728881836)
        If (feature 4 <= -73.97915649414062)
         Predict: 2268.0
        Else (feature 4 > -73.97915649414062)
         Predict: 2065.0
       Else (feature 5 > 40.752187728881836)
        Predict: 2566.0
      Else (feature 5 > 40.77285957336426)
       Predict: 3370.0
    Else (feature 3 > 40.6946907043457)
     If (feature 2 <= -73.93980407714844)
      If (feature 4 <= -73.93783187866211)
       If (feature 5 <= 40.70528793334961)
        If (feature 4 <= -74.00594329833984)
         Predict: 882.2857142857143
        Else (feature 4 > -74.00594329833984)
         Predict: 1447.1333333333334
       Else (feature 5 > 40.70528793334961)
        If (feature 5 <= 40.77510643005371)
         Predict: 728.0413043478261
        Else (feature 5 > 40.77510643005371)
         Predict: 547.6771653543307
      Else (feature 4 > -73.93783187866211)
       If (feature 5 <= 40.77285957336426)
        If (feature 5 <= 40.690895080566406)
         Predict: 2404.5
        Else (feature 5 > 40.690895080566406)
         Predict: 1424.0
       Else (feature 5 > 40.77285957336426)
        If (feature 3 <= 40.738487243652344)
         Predict: 1356.0
        Else (feature 3 > 40.738487243652344)
         Predict: 2272.5
     Else (feature 2 > -73.93980407714844)
      If (feature 3 <= 40.76089668273926)
       If (feature 0 <= 1.5)
        Predict: 604.0
       Else (feature 0 > 1.5)
        If (feature 3 <= 40.747825622558594)
         Predict: 404.0
        Else (feature 3 > 40.747825622558594)
         Predict: 771.0
      Else (feature 3 > 40.76089668273926)
       If (feature 3 <= 40.77609443664551)
        If (feature 4 <= -73.97915649414062)
         Predict: 2744.5
        Else (feature 4 > -73.97915649414062)
         Predict: 1627.0
       Else (feature 3 > 40.77609443664551)
        Predict: 854.0
  Tree 9 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 0 <= 1.5)
      If (feature 5 <= 40.74112129211426)
       If (feature 2 <= -73.97146987915039)
        Predict: 668.0
       Else (feature 2 > -73.97146987915039)
        If (feature 4 <= -73.96831512451172)
         Predict: 2223.6666666666665
        Else (feature 4 > -73.96831512451172)
         Predict: 750.5
      Else (feature 5 > 40.74112129211426)
       Predict: 2315.5
     Else (feature 0 > 1.5)
      If (feature 1 <= 3.5)
       If (feature 1 <= 1.5)
        If (feature 2 <= -73.96899795532227)
         Predict: 592.0
        Else (feature 2 > -73.96899795532227)
         Predict: 5248.0
       Else (feature 1 > 1.5)
        Predict: 3528.0
      Else (feature 1 > 3.5)
       Predict: 1884.0
    Else (feature 3 > 40.6946907043457)
     If (feature 1 <= 4.5)
      If (feature 2 <= -73.93980407714844)
       If (feature 4 <= -73.93783187866211)
        If (feature 4 <= -73.98209762573242)
         Predict: 888.4734848484849
        Else (feature 4 > -73.98209762573242)
         Predict: 670.8518518518518
       Else (feature 4 > -73.93783187866211)
        If (feature 2 <= -73.96100234985352)
         Predict: 1726.5416666666667
        Else (feature 2 > -73.96100234985352)
         Predict: 1079.6666666666667
      Else (feature 2 > -73.93980407714844)
       If (feature 3 <= 40.76089668273926)
        If (feature 4 <= -73.98209762573242)
         Predict: 836.0
        Else (feature 4 > -73.98209762573242)
         Predict: 404.0
       Else (feature 3 > 40.76089668273926)
        If (feature 1 <= 1.5)
         Predict: 1181.7
        Else (feature 1 > 1.5)
         Predict: 2328.5555555555557
     Else (feature 1 > 4.5)
      If (feature 5 <= 40.80543899536133)
       If (feature 4 <= -73.95264053344727)
        If (feature 4 <= -74.00432968139648)
         Predict: 921.8
        Else (feature 4 > -74.00432968139648)
         Predict: 492.9767441860465
       Else (feature 4 > -73.95264053344727)
        Predict: 1499.0
      Else (feature 5 > 40.80543899536133)
       Predict: 1575.0
  Tree 10 (weight 1.0):
    If (feature 4 <= -73.93783187866211)
     If (feature 3 <= 40.6946907043457)
      If (feature 0 <= 1.5)
       Predict: 2923.3571428571427
      Else (feature 0 > 1.5)
       If (feature 4 <= -73.98966598510741)
        If (feature 2 <= -73.9958724975586)
         Predict: 853.0
        Else (feature 2 > -73.9958724975586)
         Predict: 505.0
       Else (feature 4 > -73.98966598510741)
        If (feature 5 <= 40.76085090637207)
         Predict: 6187.0
        Else (feature 5 > 40.76085090637207)
         Predict: 3370.0
     Else (feature 3 > 40.6946907043457)
      If (feature 2 <= -73.93980407714844)
       If (feature 4 <= -73.98324966430664)
        If (feature 3 <= 40.752168655395515)
         Predict: 678.7321428571429
        Else (feature 3 > 40.752168655395515)
         Predict: 1002.2368421052631
       Else (feature 4 > -73.98324966430664)
        If (feature 3 <= 40.75063514709473)
         Predict: 840.1785714285714
        Else (feature 3 > 40.75063514709473)
         Predict: 577.5145228215767
      Else (feature 2 > -73.93980407714844)
       If (feature 3 <= 40.75492477416992)
        If (feature 0 <= 1.5)
         Predict: 1305.0
        Else (feature 0 > 1.5)
         Predict: 706.0
       Else (feature 3 > 40.75492477416992)
        If (feature 4 <= -73.99754333496094)
         Predict: 2860.0
        Else (feature 4 > -73.99754333496094)
         Predict: 1815.7857142857142
    Else (feature 4 > -73.93783187866211)
     If (feature 2 <= -73.96100234985352)
      If (feature 5 <= 40.77510643005371)
       If (feature 2 <= -74.00096511840819)
        If (feature 2 <= -74.00333023071289)
         Predict: 1466.0
        Else (feature 2 > -74.00333023071289)
         Predict: 1069.0
       Else (feature 2 > -74.00096511840819)
        If (feature 0 <= 1.5)
         Predict: 1985.6
        Else (feature 0 > 1.5)
         Predict: 1564.5714285714287
      Else (feature 5 > 40.77510643005371)
       If (feature 2 <= -74.00096511840819)
        Predict: 1003.0
       Else (feature 2 > -74.00096511840819)
        If (feature 3 <= 40.75492477416992)
         Predict: 2534.0
        Else (feature 3 > 40.75492477416992)
         Predict: 2396.0
     Else (feature 2 > -73.96100234985352)
      If (feature 5 <= 40.690895080566406)
       If (feature 4 <= -73.87130355834961)
        Predict: 256.0
       Else (feature 4 > -73.87130355834961)
        Predict: 3006.5
      Else (feature 5 > 40.690895080566406)
       If (feature 1 <= 2.5)
        If (feature 2 <= -73.8634262084961)
         Predict: 1260.3333333333333
        Else (feature 2 > -73.8634262084961)
         Predict: 890.25
       Else (feature 1 > 2.5)
        If (feature 2 <= -73.8634262084961)
         Predict: 1694.0
        Else (feature 2 > -73.8634262084961)
         Predict: 1777.0
  Tree 11 (weight 1.0):
    If (feature 4 <= -73.87130355834961)
     If (feature 3 <= 40.6946907043457)
      If (feature 5 <= 40.752187728881836)
       If (feature 1 <= 2.5)
        If (feature 5 <= 40.70528793334961)
         Predict: 901.4444444444445
        Else (feature 5 > 40.70528793334961)
         Predict: 1273.3333333333333
       Else (feature 1 > 2.5)
        If (feature 4 <= -73.99144744873047)
         Predict: 2453.0
        Else (feature 4 > -73.99144744873047)
         Predict: 1974.5
      Else (feature 5 > 40.752187728881836)
       If (feature 0 <= 1.5)
        Predict: 5109.0
       Else (feature 0 > 1.5)
        Predict: 5623.6
     Else (feature 3 > 40.6946907043457)
      If (feature 3 <= 40.77609443664551)
       If (feature 3 <= 40.772178649902344)
        If (feature 4 <= -73.93783187866211)
         Predict: 757.0037950664137
        Else (feature 4 > -73.93783187866211)
         Predict: 1333.3529411764705
       Else (feature 3 > 40.772178649902344)
        If (feature 2 <= -73.93980407714844)
         Predict: 767.28
        Else (feature 2 > -73.93980407714844)
         Predict: 2062.875
      Else (feature 3 > 40.77609443664551)
       If (feature 4 <= -73.98495483398438)
        If (feature 5 <= 40.762474060058594)
         Predict: 1035.0
        Else (feature 5 > 40.762474060058594)
         Predict: 1780.8333333333333
       Else (feature 4 > -73.98495483398438)
        If (feature 1 <= 1.5)
         Predict: 556.2096774193549
        Else (feature 1 > 1.5)
         Predict: 394.70588235294116
    Else (feature 4 > -73.87130355834961)
     If (feature 3 <= 40.78046417236328)
      If (feature 1 <= 2.5)
       If (feature 5 <= 40.690895080566406)
        If (feature 2 <= -73.95446014404297)
         Predict: 2402.1428571428573
        Else (feature 2 > -73.95446014404297)
         Predict: 580.0
       Else (feature 5 > 40.690895080566406)
        If (feature 2 <= -73.97738647460938)
         Predict: 1328.5
        Else (feature 2 > -73.97738647460938)
         Predict: 1619.8333333333333
      Else (feature 1 > 2.5)
       If (feature 0 <= 1.5)
        Predict: 2344.75
       Else (feature 0 > 1.5)
        Predict: 3528.0
     Else (feature 3 > 40.78046417236328)
      Predict: 1403.0
  Tree 12 (weight 1.0):
    If (feature 2 <= -73.8634262084961)
     If (feature 4 <= -73.93783187866211)
      If (feature 5 <= 40.71343803405762)
       If (feature 2 <= -73.99901962280273)
        If (feature 5 <= 40.690895080566406)
         Predict: 1486.0
        Else (feature 5 > 40.690895080566406)
         Predict: 579.4285714285714
       Else (feature 2 > -73.99901962280273)
        If (feature 2 <= -73.93980407714844)
         Predict: 1183.9347826086957
        Else (feature 2 > -73.93980407714844)
         Predict: 2753.0
      Else (feature 5 > 40.71343803405762)
       If (feature 2 <= -74.00597763061523)
        If (feature 4 <= -73.99144744873047)
         Predict: 468.6666666666667
        Else (feature 4 > -73.99144744873047)
         Predict: 1212.2307692307693
       Else (feature 2 > -74.00597763061523)
        If (feature 2 <= -73.93980407714844)
         Predict: 670.1724738675958
        Else (feature 2 > -73.93980407714844)
         Predict: 1268.2857142857142
     Else (feature 4 > -73.93783187866211)
      If (feature 2 <= -73.95446014404297)
       If (feature 5 <= 40.690895080566406)
        Predict: 2485.0
       Else (feature 5 > 40.690895080566406)
        If (feature 5 <= 40.77510643005371)
         Predict: 1614.909090909091
        Else (feature 5 > 40.77510643005371)
         Predict: 981.0
      Else (feature 2 > -73.95446014404297)
       If (feature 4 <= -73.87130355834961)
        If (feature 2 <= -73.93980407714844)
         Predict: 256.0
        Else (feature 2 > -73.93980407714844)
         Predict: 704.0
       Else (feature 4 > -73.87130355834961)
        Predict: 1059.0
    Else (feature 2 > -73.8634262084961)
     If (feature 4 <= -73.98495483398438)
      If (feature 5 <= 40.73106002807617)
       Predict: 2453.0
      Else (feature 5 > 40.73106002807617)
       If (feature 5 <= 40.76085090637207)
        Predict: 6187.0
       Else (feature 5 > 40.76085090637207)
        Predict: 5109.0
     Else (feature 4 > -73.98495483398438)
      If (feature 4 <= -73.97111511230469)
       If (feature 0 <= 1.5)
        Predict: 2048.75
       Else (feature 0 > 1.5)
        If (feature 4 <= -73.9727897644043)
         Predict: 2819.0
        Else (feature 4 > -73.9727897644043)
         Predict: 1884.0
      Else (feature 4 > -73.97111511230469)
       If (feature 1 <= 1.5)
        Predict: 1108.4
       Else (feature 1 > 1.5)
        Predict: 1777.0
  Tree 13 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 2 <= -73.97146987915039)
      If (feature 2 <= -73.9958724975586)
       Predict: 853.0
      Else (feature 2 > -73.9958724975586)
       Predict: 668.0
     Else (feature 2 > -73.97146987915039)
      If (feature 5 <= 40.752187728881836)
       If (feature 5 <= 40.73424530029297)
        If (feature 4 <= -73.9808578491211)
         Predict: 2044.6666666666667
        Else (feature 4 > -73.9808578491211)
         Predict: 3176.0
       Else (feature 5 > 40.73424530029297)
        If (feature 5 <= 40.74112129211426)
         Predict: 1349.0
        Else (feature 5 > 40.74112129211426)
         Predict: 2065.0
      Else (feature 5 > 40.752187728881836)
       If (feature 5 <= 40.762474060058594)
        If (feature 0 <= 1.5)
         Predict: 5109.0
        Else (feature 0 > 1.5)
         Predict: 6187.0
       Else (feature 5 > 40.762474060058594)
        If (feature 4 <= -73.9808578491211)
         Predict: 2566.0
        Else (feature 4 > -73.9808578491211)
         Predict: 3370.0
    Else (feature 3 > 40.6946907043457)
     If (feature 2 <= -73.93980407714844)
      If (feature 5 <= 40.70528793334961)
       If (feature 2 <= -73.9958724975586)
        If (feature 1 <= 1.5)
         Predict: 1128.25
        Else (feature 1 > 1.5)
         Predict: 638.25
       Else (feature 2 > -73.9958724975586)
        If (feature 3 <= 40.734697341918945)
         Predict: 1197.1666666666667
        Else (feature 3 > 40.734697341918945)
         Predict: 2072.0
      Else (feature 5 > 40.70528793334961)
       If (feature 3 <= 40.71293067932129)
        If (feature 5 <= 40.74869728088379)
         Predict: 676.75
        Else (feature 5 > 40.74869728088379)
         Predict: 1774.75
       Else (feature 3 > 40.71293067932129)
        If (feature 2 <= -73.97544479370117)
         Predict: 799.371158392435
        Else (feature 2 > -73.97544479370117)
         Predict: 677.8122065727699
     Else (feature 2 > -73.93980407714844)
      If (feature 5 <= 40.74338722229004)
       If (feature 4 <= -74.00594329833984)
        Predict: 2753.0
       Else (feature 4 > -74.00594329833984)
        If (feature 4 <= -73.99754333496094)
         Predict: 1305.0
        Else (feature 4 > -73.99754333496094)
         Predict: 776.3333333333334
      Else (feature 5 > 40.74338722229004)
       If (feature 2 <= -73.8634262084961)
        If (feature 1 <= 3.5)
         Predict: 1946.7
        Else (feature 1 > 3.5)
         Predict: 1330.75
       Else (feature 2 > -73.8634262084961)
        If (feature 5 <= 40.74869728088379)
         Predict: 2817.0
        Else (feature 5 > 40.74869728088379)
         Predict: 2699.0
  Tree 14 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 2 <= -73.96899795532227)
      If (feature 4 <= -74.00432968139648)
       Predict: 853.0
      Else (feature 4 > -74.00432968139648)
       Predict: 505.0
     Else (feature 2 > -73.96899795532227)
      If (feature 4 <= -73.9727897644043)
       If (feature 0 <= 1.5)
        If (feature 5 <= 40.73424530029297)
         Predict: 2114.0
        Else (feature 5 > 40.73424530029297)
         Predict: 5109.0
       Else (feature 0 > 1.5)
        If (feature 4 <= -73.98495483398438)
         Predict: 4320.0
        Else (feature 4 > -73.98495483398438)
         Predict: 3002.6666666666665
      Else (feature 4 > -73.9727897644043)
       If (feature 1 <= 1.5)
        Predict: 2088.25
       Else (feature 1 > 1.5)
        If (feature 1 <= 3.5)
         Predict: 2944.3333333333335
        Else (feature 1 > 3.5)
         Predict: 1884.0
    Else (feature 3 > 40.6946907043457)
     If (feature 5 <= 40.70528793334961)
      If (feature 2 <= -73.99408340454102)
       If (feature 3 <= 40.734697341918945)
        If (feature 4 <= -73.98209762573242)
         Predict: 866.875
        Else (feature 4 > -73.98209762573242)
         Predict: 1069.0
       Else (feature 3 > 40.734697341918945)
        Predict: 586.0
      Else (feature 2 > -73.99408340454102)
       If (feature 3 <= 40.734697341918945)
        If (feature 0 <= 1.5)
         Predict: 847.0
        Else (feature 0 > 1.5)
         Predict: 1390.5
       Else (feature 3 > 40.734697341918945)
        If (feature 4 <= -74.0104751586914)
         Predict: 1172.0
        Else (feature 4 > -74.0104751586914)
         Predict: 2293.5
     Else (feature 5 > 40.70528793334961)
      If (feature 4 <= -73.93783187866211)
       If (feature 2 <= -73.93980407714844)
        If (feature 3 <= 40.71845626831055)
         Predict: 1065.8235294117646
        Else (feature 3 > 40.71845626831055)
         Predict: 725.6954954954955
       Else (feature 2 > -73.93980407714844)
        If (feature 1 <= 4.5)
         Predict: 2151.4285714285716
        Else (feature 1 > 4.5)
         Predict: 995.6666666666666
      Else (feature 4 > -73.93783187866211)
       If (feature 2 <= -73.96899795532227)
        If (feature 2 <= -73.97544479370117)
         Predict: 1632.7777777777778
        Else (feature 2 > -73.97544479370117)
         Predict: 2176.125
       Else (feature 2 > -73.96899795532227)
        If (feature 3 <= 40.747825622558594)
         Predict: 662.6666666666666
        Else (feature 3 > 40.747825622558594)
         Predict: 1049.9
  Tree 15 (weight 1.0):
    If (feature 4 <= -73.93783187866211)
     If (feature 4 <= -73.98209762573242)
      If (feature 2 <= -73.8634262084961)
       If (feature 2 <= -73.9809341430664)
        If (feature 1 <= 1.5)
         Predict: 733.3396226415094
        Else (feature 1 > 1.5)
         Predict: 639.9298245614035
       Else (feature 2 > -73.9809341430664)
        If (feature 2 <= -73.96899795532227)
         Predict: 971.1428571428571
        Else (feature 2 > -73.96899795532227)
         Predict: 1268.7297297297298
      Else (feature 2 > -73.8634262084961)
       If (feature 4 <= -74.00594329833984)
        Predict: 1733.0
       Else (feature 4 > -74.00594329833984)
        Predict: 4609.833333333333
     Else (feature 4 > -73.98209762573242)
      If (feature 3 <= 40.75063514709473)
       If (feature 2 <= -73.95005798339844)
        If (feature 4 <= -73.9727897644043)
         Predict: 730.1090909090909
        Else (feature 4 > -73.9727897644043)
         Predict: 1161.8048780487804
       Else (feature 2 > -73.95005798339844)
        If (feature 0 <= 1.5)
         Predict: 2566.0
        Else (feature 0 > 1.5)
         Predict: 1884.0
      Else (feature 3 > 40.75063514709473)
       If (feature 2 <= -73.93980407714844)
        If (feature 4 <= -73.96831512451172)
         Predict: 637.6213592233009
        Else (feature 4 > -73.96831512451172)
         Predict: 517.0204081632653
       Else (feature 2 > -73.93980407714844)
        If (feature 5 <= 40.758445739746094)
         Predict: 1607.6666666666667
        Else (feature 5 > 40.758445739746094)
         Predict: 2230.0
    Else (feature 4 > -73.93783187866211)
     If (feature 5 <= 40.73819923400879)
      If (feature 0 <= 1.5)
       If (feature 3 <= 40.734697341918945)
        If (feature 1 <= 1.5)
         Predict: 1269.3333333333333
        Else (feature 1 > 1.5)
         Predict: 1777.0
       Else (feature 3 > 40.734697341918945)
        If (feature 2 <= -73.99187088012695)
         Predict: 2743.0
        Else (feature 2 > -73.99187088012695)
         Predict: 2066.0
      Else (feature 0 > 1.5)
       If (feature 5 <= 40.690895080566406)
        Predict: 2485.0
       Else (feature 5 > 40.690895080566406)
        Predict: 1784.0
     Else (feature 5 > 40.73819923400879)
      If (feature 5 <= 40.74869728088379)
       If (feature 2 <= -73.8634262084961)
        If (feature 5 <= 40.74338722229004)
         Predict: 604.0
        Else (feature 5 > 40.74338722229004)
         Predict: 404.0
       Else (feature 2 > -73.8634262084961)
        Predict: 921.0
      Else (feature 5 > 40.74869728088379)
       If (feature 5 <= 40.77285957336426)
        If (feature 1 <= 2.5)
         Predict: 1354.0
        Else (feature 1 > 2.5)
         Predict: 598.0
       Else (feature 5 > 40.77285957336426)
        If (feature 1 <= 1.5)
         Predict: 1265.0
        Else (feature 1 > 1.5)
         Predict: 2079.5
  Tree 16 (weight 1.0):
    If (feature 1 <= 4.5)
     If (feature 1 <= 1.5)
      If (feature 4 <= -73.93783187866211)
       If (feature 5 <= 40.71343803405762)
        If (feature 4 <= -74.0104751586914)
         Predict: 771.0
        Else (feature 4 > -74.0104751586914)
         Predict: 1616.7826086956522
       Else (feature 5 > 40.71343803405762)
        If (feature 4 <= -73.96831512451172)
         Predict: 884.0228013029316
        Else (feature 4 > -73.96831512451172)
         Predict: 599.0491803278688
      Else (feature 4 > -73.93783187866211)
       If (feature 3 <= 40.71293067932129)
        If (feature 5 <= 40.690895080566406)
         Predict: 418.0
        Else (feature 5 > 40.690895080566406)
         Predict: 990.0
       Else (feature 3 > 40.71293067932129)
        If (feature 5 <= 40.72003936767578)
         Predict: 2257.2
        Else (feature 5 > 40.72003936767578)
         Predict: 1486.05
     Else (feature 1 > 1.5)
      If (feature 4 <= -73.93783187866211)
       If (feature 3 <= 40.79401969909668)
        If (feature 5 <= 40.690895080566406)
         Predict: 1604.142857142857
        Else (feature 5 > 40.690895080566406)
         Predict: 668.0364963503649
       Else (feature 3 > 40.79401969909668)
        If (feature 4 <= -73.99471282958984)
         Predict: 1105.0
        Else (feature 4 > -73.99471282958984)
         Predict: 1628.0
      Else (feature 4 > -73.93783187866211)
       If (feature 5 <= 40.77285957336426)
        If (feature 3 <= 40.75492477416992)
         Predict: 1752.5
        Else (feature 3 > 40.75492477416992)
         Predict: 1000.5
       Else (feature 5 > 40.77285957336426)
        If (feature 3 <= 40.75492477416992)
         Predict: 2534.0
        Else (feature 3 > 40.75492477416992)
         Predict: 1836.6
    Else (feature 1 > 4.5)
     If (feature 3 <= 40.6946907043457)
      Predict: 2453.0
     Else (feature 3 > 40.6946907043457)
      If (feature 2 <= -73.9864501953125)
       If (feature 2 <= -74.00096511840819)
        If (feature 3 <= 40.740970611572266)
         Predict: 509.5
        Else (feature 3 > 40.740970611572266)
         Predict: 819.0
       Else (feature 2 > -74.00096511840819)
        If (feature 1 <= 5.5)
         Predict: 214.75
        Else (feature 1 > 5.5)
         Predict: 478.0
      Else (feature 2 > -73.9864501953125)
       If (feature 3 <= 40.766544342041016)
        If (feature 2 <= -73.96899795532227)
         Predict: 663.7826086956521
        Else (feature 2 > -73.96899795532227)
         Predict: 1499.0
       Else (feature 3 > 40.766544342041016)
        If (feature 2 <= -73.96648406982422)
         Predict: 237.0
        Else (feature 2 > -73.96648406982422)
         Predict: 534.0
  Tree 17 (weight 1.0):
    If (feature 1 <= 2.5)
     If (feature 2 <= -73.8634262084961)
      If (feature 5 <= 40.70528793334961)
       If (feature 2 <= -73.98375701904297)
        If (feature 4 <= -73.87130355834961)
         Predict: 1333.2222222222222
        Else (feature 4 > -73.87130355834961)
         Predict: 2517.3333333333335
       Else (feature 2 > -73.98375701904297)
        If (feature 4 <= -73.9727897644043)
         Predict: 740.7142857142857
        Else (feature 4 > -73.9727897644043)
         Predict: 1399.5
      Else (feature 5 > 40.70528793334961)
       If (feature 2 <= -73.93980407714844)
        If (feature 3 <= 40.71845626831055)
         Predict: 926.6744186046511
        Else (feature 3 > 40.71845626831055)
         Predict: 705.1409978308026
       Else (feature 2 > -73.93980407714844)
        If (feature 3 <= 40.76089668273926)
         Predict: 813.4285714285714
        Else (feature 3 > 40.76089668273926)
         Predict: 1628.6666666666667
     Else (feature 2 > -73.8634262084961)
      If (feature 4 <= -73.9727897644043)
       If (feature 0 <= 1.5)
        Predict: 1733.0
       Else (feature 0 > 1.5)
        Predict: 4013.125
      Else (feature 4 > -73.9727897644043)
       If (feature 4 <= -73.96572494506836)
        Predict: 1219.0
       Else (feature 4 > -73.96572494506836)
        If (feature 3 <= 40.6946907043457)
         Predict: 1182.0
        Else (feature 3 > 40.6946907043457)
         Predict: 946.0
    Else (feature 1 > 2.5)
     If (feature 4 <= -73.87130355834961)
      If (feature 5 <= 40.765913009643555)
       If (feature 3 <= 40.6946907043457)
        If (feature 5 <= 40.73106002807617)
         Predict: 2453.0
        Else (feature 5 > 40.73106002807617)
         Predict: 1929.25
       Else (feature 3 > 40.6946907043457)
        If (feature 2 <= -73.8634262084961)
         Predict: 755.1071428571429
        Else (feature 2 > -73.8634262084961)
         Predict: 2746.2
      Else (feature 5 > 40.765913009643555)
       If (feature 3 <= 40.740970611572266)
        If (feature 5 <= 40.80543899536133)
         Predict: 966.0
        Else (feature 5 > 40.80543899536133)
         Predict: 1280.0
       Else (feature 3 > 40.740970611572266)
        If (feature 1 <= 3.5)
         Predict: 695.3636363636364
        Else (feature 1 > 3.5)
         Predict: 481.3157894736842
     Else (feature 4 > -73.87130355834961)
      If (feature 0 <= 1.5)
       If (feature 5 <= 40.73819923400879)
        Predict: 1777.0
       Else (feature 5 > 40.73819923400879)
        Predict: 2534.0
      Else (feature 0 > 1.5)
       Predict: 3528.0
  Tree 18 (weight 1.0):
    If (feature 3 <= 40.6946907043457)
     If (feature 2 <= -73.97146987915039)
      If (feature 2 <= -73.9958724975586)
       Predict: 853.0
      Else (feature 2 > -73.9958724975586)
       Predict: 668.0
     Else (feature 2 > -73.97146987915039)
      If (feature 4 <= -73.98495483398438)
       If (feature 5 <= 40.73424530029297)
        If (feature 5 <= 40.73106002807617)
         Predict: 2453.0
        Else (feature 5 > 40.73106002807617)
         Predict: 2114.0
       Else (feature 5 > 40.73424530029297)
        Predict: 5109.0
      Else (feature 4 > -73.98495483398438)
       If (feature 4 <= -73.96831512451172)
        Predict: 2485.0
       Else (feature 4 > -73.96831512451172)
        If (feature 1 <= 1.5)
         Predict: 1496.3333333333333
        Else (feature 1 > 1.5)
         Predict: 3528.0
    Else (feature 3 > 40.6946907043457)
     If (feature 2 <= -73.93980407714844)
      If (feature 5 <= 40.70528793334961)
       If (feature 1 <= 2.5)
        If (feature 3 <= 40.74964714050293)
         Predict: 960.1
        Else (feature 3 > 40.74964714050293)
         Predict: 1885.0
       Else (feature 1 > 2.5)
        If (feature 4 <= -74.0104751586914)
         Predict: 1093.0
        Else (feature 4 > -74.0104751586914)
         Predict: 795.0
      Else (feature 5 > 40.70528793334961)
       If (feature 4 <= -73.93783187866211)
        If (feature 1 <= 1.5)
         Predict: 770.9277389277389
        Else (feature 1 > 1.5)
         Predict: 640.1
       Else (feature 4 > -73.93783187866211)
        If (feature 3 <= 40.743438720703125)
         Predict: 1397.0
        Else (feature 3 > 40.743438720703125)
         Predict: 1819.0
     Else (feature 2 > -73.93980407714844)
      If (feature 4 <= -73.99754333496094)
       Predict: 2967.0
      Else (feature 4 > -73.99754333496094)
       If (feature 3 <= 40.77609443664551)
        If (feature 4 <= -73.93783187866211)
         Predict: 1875.4166666666667
        Else (feature 4 > -73.93783187866211)
         Predict: 1047.4285714285713
       Else (feature 3 > 40.77609443664551)
        If (feature 0 <= 1.5)
         Predict: 810.0
        Else (feature 0 > 1.5)
         Predict: 894.4
  Tree 19 (weight 1.0):
    If (feature 5 <= 40.690895080566406)
     If (feature 1 <= 2.5)
      If (feature 4 <= -73.97111511230469)
       If (feature 3 <= 40.6946907043457)
        If (feature 4 <= -74.00594329833984)
         Predict: 1733.0
        Else (feature 4 > -74.00594329833984)
         Predict: 613.6666666666666
       Else (feature 3 > 40.6946907043457)
        If (feature 2 <= -73.9958724975586)
         Predict: 1048.0
        Else (feature 2 > -73.9958724975586)
         Predict: 1396.3333333333333
      Else (feature 4 > -73.97111511230469)
       If (feature 3 <= 40.72142791748047)
        If (feature 4 <= -73.94768524169922)
         Predict: 1485.0
        Else (feature 4 > -73.94768524169922)
         Predict: 418.0
       Else (feature 3 > 40.72142791748047)
        If (feature 3 <= 40.75752258300781)
         Predict: 2588.2
        Else (feature 3 > 40.75752258300781)
         Predict: 2120.0
     Else (feature 1 > 2.5)
      Predict: 3528.0
    Else (feature 5 > 40.690895080566406)
     If (feature 0 <= 1.5)
      If (feature 2 <= -73.93980407714844)
       If (feature 2 <= -73.95605087280273)
        If (feature 4 <= -73.93783187866211)
         Predict: 727.3877551020408
        Else (feature 4 > -73.93783187866211)
         Predict: 2035.8
       Else (feature 2 > -73.95605087280273)
        If (feature 1 <= 1.5)
         Predict: 743.64
        Else (feature 1 > 1.5)
         Predict: 508.35714285714283
      Else (feature 2 > -73.93980407714844)
       If (feature 5 <= 40.74338722229004)
        If (feature 5 <= 40.73424530029297)
         Predict: 1948.0
        Else (feature 5 > 40.73424530029297)
         Predict: 1017.25
       Else (feature 5 > 40.74338722229004)
        If (feature 3 <= 40.6946907043457)
         Predict: 3512.714285714286
        Else (feature 3 > 40.6946907043457)
         Predict: 2004.5833333333333
     Else (feature 0 > 1.5)
      If (feature 4 <= -73.93783187866211)
       If (feature 3 <= 40.71293067932129)
        If (feature 5 <= 40.752187728881836)
         Predict: 1230.7777777777778
        Else (feature 5 > 40.752187728881836)
         Predict: 3049.0
       Else (feature 3 > 40.71293067932129)
        If (feature 4 <= -74.00090789794922)
         Predict: 835.1698113207547
        Else (feature 4 > -74.00090789794922)
         Predict: 662.6530612244898
      Else (feature 4 > -73.93783187866211)
       If (feature 3 <= 40.77383041381836)
        If (feature 5 <= 40.77510643005371)
         Predict: 1516.7857142857142
        Else (feature 5 > 40.77510643005371)
         Predict: 2228.0
       Else (feature 3 > 40.77383041381836)
        If (feature 1 <= 1.5)
         Predict: 1266.0
        Else (feature 1 > 1.5)
         Predict: 1403.0

accuracy: 4845.3933640859
----------------- accuracy ----------------- 
...

```



```bash

# run KNN via pyspark 
# https://github.com/yennanliu/utility_shell/blob/master/spark/launch_pyspark.sh
$ source activate pyspark_ # launch ur pyspark dev env 
$ export SPARK_HOME=/Users/$USER/spark
$ export PATH=$SPARK_HOME/bin:$PATH
$ spark-submit  train_spark_KNN.py


```