
1. count.dat:

This is the raw output from an XSPEC session. It contains 4 blocks
of data, one for each spectrum. Each block has 5 columns

1 wavelength (Angstrom) 
2 +- wavelength size of the bin (Angstrom)
3 spectrum in counts/Angstrom
4 error/Angstrom where error is conts**0.5 (this is not needed)
5 best-fit model in counts/Angstrom.

The XSPEC model is summarized here:

===================================================================

4 files 4 spectra 
Spectrum 1  Spectral Data File: pg1116-oldobs_new_leg_abs1.pha
Net count rate (cts/s) for Spectrum:1  1.618e-02 +/- 4.287e-04
 Assigned to Data Group 1 and Plot Group 1
  Noticed Channels:  14706-14864
  Telescope: CHANDRA Instrument: HRC  Channel Type: PI
  Exposure Time: 8.805e+04 sec
 Using fit statistic: cstat
 Using Response (RMF) File            pg1116-oldobs_new_leg_abs1.rmf for Source 1
 Using Auxiliary Response (ARF) File  pg1116-oldobs_new_leg_abs1.arf

 Spectral data counts: 1425
 Model predicted rate: 1.61848E-02

Spectrum 2  Spectral Data File: pg1116-newobs_new_leg_abs1.pha
Net count rate (cts/s) for Spectrum:2  1.825e-02 +/- 2.613e-04
 Assigned to Data Group 2 and Plot Group 2
  Noticed Channels:  14706-14864
  Telescope: CHANDRA Instrument: HRC  Channel Type: PI
  Exposure Time: 2.674e+05 sec
 Using fit statistic: cstat
 Using Response (RMF) File            pg1116-newobs_new_leg_abs1.rmf for Source 1
 Using Auxiliary Response (ARF) File  pg1116-newobs_new_leg_abs1.arf

 Spectral data counts: 4882
 Model predicted rate: 1.82547E-02

Spectrum 3  Spectral Data File: pg1116-newobs_new_leg_abs1_bkg.pha
Net count rate (cts/s) for Spectrum:3  1.323e-01 +/- 7.032e-04
 Assigned to Data Group 3 and Plot Group 3
  Noticed Channels:  14706-14864
  Telescope: CHANDRA Instrument: HRC  Channel Type: PI
  Exposure Time: 2.675e+05 sec
 Using fit statistic: cstat
 Using Response (RMF) File            pg1116-newobs_new_leg_abs1.rmf for Source 2

 Spectral data counts: 35384
 Model predicted rate: 0.132275

Spectrum 4  Spectral Data File: pg1116-oldobs_new_leg_abs1_bkg.pha
Net count rate (cts/s) for Spectrum:4  4.962e-02 +/- 7.507e-04
 Assigned to Data Group 4 and Plot Group 4
  Noticed Channels:  14706-14864
  Telescope: CHANDRA Instrument: HRC  Channel Type: PI
  Exposure Time: 8.805e+04 sec
 Using fit statistic: cstat
 Using Response (RMF) File            pg1116-oldobs_new_leg_abs1.rmf for Source 2

 Spectral data counts: 4369
 Model predicted rate: 4.96221E-02

Current model list:

========================================================================

Model powerlaw<1> + powerlaw<2> Source No.: 1   Active/On
Model Model Component  Parameter  Unit     Value
 par  comp
                           Data group: 1
   1    1   powerlaw   PhoIndex            2.00000      frozen
   2    1   powerlaw   norm                4.88595E-03  +/-  1.89517E-04  
   3    2   powerlaw   PhoIndex            2.00000      frozen
   4    2   powerlaw   norm                2.16091E-03  = myback:p4/16.565/10
                           Data group: 2
   5    1   powerlaw   PhoIndex            2.00000      = p1
   6    1   powerlaw   norm                2.59496E-03  +/-  1.39710E-04  
   7    2   powerlaw   PhoIndex            2.00000      frozen
   8    2   powerlaw   norm                6.83119E-03  = myback:p2/13.968/10
________________________________________________________________________


========================================================================
Model myback:powerlaw<1> Source No.: 2   Active/On
Model Model Component  Parameter  Unit     Value
 par  comp
                           Data group: 3
   1    1   powerlaw   PhoIndex            2.00000      frozen
   2    1   powerlaw   norm                0.954181     +/-  5.07256E-03  
                           Data group: 4
   3    1   powerlaw   PhoIndex            2.00000      = myback:p1
   4    1   powerlaw   norm                0.357955     +/-  5.41548E-03  
________________________________________________________________________


Fit statistic  : C-Statistic                  191.87     using 159 bins, spectrum 1, group 1.
                 C-Statistic                  170.40     using 159 bins, spectrum 2, group 2.
                 C-Statistic                  153.46     using 159 bins, spectrum 3, group 3.
                 C-Statistic                  171.35     using 159 bins, spectrum 4, group 4.
Total fit statistic                           687.08     with 632 d.o.f.

=====================================================================

The model setup is really hard to read in XSPEC, but it goes like this:

SOURCE: The first block is for the SOURCE of OLD and NEW observations.
The models are:

1. a power-law for the SOURCE (parameters 1-2 and 5-6, respectively for OLD and NEW obs.)
2. a power-law for the BACK (parameters 3-4 and 7-8, respectively for OLD and NEW obs.)

Power-law indices are fixed at 2,  but they can be left free to add another parameter, so currently each model has only the
normalization free. The parameters for the BACK are obtained from a fit to the BACK.

BACK: The second block is for the BACK of NEW and OLD observations, which have
as a model a power-law (index fixed at 2, it can be left free)

1. a power law for the BACK NEW (parameters myback:1-2)
1. a power law for the BACK OLD (parameters myback:3-4)

Notice how the best-fit power-law models for BACK are rescaled when applied to the source, by 
two factors (myback:p4/16.565/10 and  myback:p2/13.968/10) that are deterministic.
There is a brief explanation for them in the python codes too. You should not need to worry about them.

2. countFormat0.dat through countFormat3.dat:

A reformatted version of count.dat, for datasets 1-4, with wavelengths, counts and best-fit model
for each bin (159 bins in each spectrum). This is generated by 
the script pg1116Spectra.py, which uses the function in StatFunctions.py.
So we can generate the formatted spectra again, when there are changes
in the XSPEC session in the future. These are the spectra that follow the Poisson count model.

countFormat0.dat: OLD SOURCE (wavelength, counts, model)
countFormat1.dat: NEW SOURCE (same) 
countFormat2.dat: NEW BACK   (same)
countFormat3.dat: OLD BACK   (same)


3. pg1116Spectra.py, StatFunctions.py: python scripts
The scripts reproduce the same cmin that XSPEC returns, so that we know
exactly what XSPEC does. 
