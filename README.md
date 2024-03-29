# Paper
See **[Writer-dependent Off-line Signature Verification with Neural Networks](https://github.com/KaihangFu/Signature-Verification-System/blob/master/Writer-dependent%20Off-line%20Signature%20Verification%20with%20Neural%20Networks.pdf)**

# Signature-Verification-System
Extract a set of features from two signature images, then use neural network as classifier to decide if two signatures are written by the same writer.

# Introduction
Define a writer-dependent off-line signature verification system.

Design a set of 56 features of a signature image, including 8 global features and 48 local features.

Accordding to experiment in ICDAR 2011 SigComp database and GPDS database, I get a really high test accuracy that higher than 90%.

This signature verification method only need a small set of sigantures as train data, such as 20 or 30 or 40 geneuine signatures and 10 or 20 forged signatures, and these signature are easy to collect, we can let the geneuine writer to write geneuine signatures that we need and let computer program or others skilled writer to produce some skilled forged sigantures.

This signature verification method is really fast to make decision and the system is really ingenious, simple and accurate enough.

# Verification Framework
See **[Writer-dependent-SV.pdf](https://github.com/KaihangFu/Signature-Verification-System/blob/master/Writer-dependent-SV.pdf)**

# Experiment Project
Totally 10 projects(for 10 test geneuine writers),user1 means writer1, user2 means writer2, ... ,etc.

Every project includes a python code document(neural network clssifier), 5 traindata.csv and 5 testdata.csv documents, testdataW.csv means worst skilled forgeries-writer signature test experiment testdata(you can see Writer-dependent-SV.pdf), testdataM.csv means middle skilled forgeries-writer, testdataB.csv means best skilled forgeries-writer, testdataR.csv means randomly selected skilled forgeries-writer, testdataA.csv means all skilled forgeries-writer signature test experiment testdata, similarly the traindata.

These csv files have 57 columns, representing 56 features and 1 label(the same(1) or different(0) writers).

Experiment result can be seen in Expeiment-Test-Result-Data.pdf.

Another experiment project is carried on GPDS Synthetic Signature corpus, I use first 100 individuals' signatures to do experiment.

# Feature Extraction
In matlab, see Feature Extraction folder.

Before run the code, you must edit the code according your choice.

# Database
ICDAR 2011 SigComp URL: http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)

In experiment I only use off-line Dutch training set.

GPDS corpus URL: http://www.gpds.ulpgc.es

I only use first 100 individuals' signatures.
