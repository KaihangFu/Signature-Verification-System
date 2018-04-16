# Signature-Verification-System
Extract a set of features from two signature images, then use neural network as classifier to decide if two signatures are written by the same writer.
# Introduction
Define a writer-dependent off-line signature verification system.

Design a set of 56 features of a signature image, including 8 global features and 48 local features.

Accordding to experiment in ICDAR 2011 SigComp database, I get a really high test accuracy that higher than 90%.

This signature verification method only need a small set of sigantures as train data, such as 20 or 30 or 40 geneuine signatures and 10 or 20 forged signatures, and these signature are easy to collect, we can let the geneuine writer to write geneuine signatures that we need and let computer program or others skilled writer to produce some skilled forged sigantures.

This signature verification method is really fast to make decision and the system is really ingenious, simple and accurate enough.
# Verification Framework
See Writer-dependent-SV.pdf
# Run the code
