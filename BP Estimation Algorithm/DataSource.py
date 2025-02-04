#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from enum import Enum


# In[ ]:

# Class containing all of the data collection records being used to train the algorithm
# These records can be found in the folder "JSONData"

class DataSource(Enum):
    rec_212_T = "rec_212_T"
    rec_44_T = "rec_44_T"
    rec_159_T = "rec_159_T"
    rec_134_T = "rec_134_T"
    rec_358_T = "rec_358_T"
    rec_500_T = "rec_500_T"
    rec_18_T = "rec_18_T"
    rec_411_T = "rec_411_T"
    rec_152_T = "rec_152_T"
    rec_327_T = "rec_327_T"
    rec_240_T = "rec_240_T"
    rec_431_T = "rec_431_T"
    rec_57_T = "rec_57_T"
    rec_157_T = "rec_157_T"
    rec_408_T = "rec_408_T"
    rec_390_T = "rec_390_T"
    rec_229_T = "rec_229_T"
    rec_401_T = "rec_401_T"
    rec_489_T = "rec_489_T"
    rec_285_T = "rec_285_T"
    rec_384_T = "rec_384_T"
    rec_447_T = "rec_447_T"
    rec_260_T = "rec_260_T"
    rec_193_T = "rec_193_T"
    rec_371_T = "rec_371_T"
    rec_479_T = "rec_479_T"
    rec_243_T = "rec_243_T"
    rec_189_T = "rec_189_T"
    rec_405_T = "rec_405_T"
    rec_67_T = "rec_67_T"
    rec_306_T = "rec_306_T"
    rec_21_T = "rec_21_T"
    rec_204_T = "rec_204_T"
    rec_158_T = "rec_158_T"
    rec_223_T = "rec_223_T"
    rec_257_T = "rec_257_T"
    rec_245_T = "rec_245_T"
    rec_407_T = "rec_407_T"
    rec_40_T = "rec_40_T"
    rec_1_T = "rec_1_T"
    rec_284_T = "rec_284_T"
    rec_185_T = "rec_185_T"
    rec_108_T = "rec_108_T"
    rec_25_T = "rec_25_T"
    rec_32_T = "rec_32_T"
    rec_343_T = "rec_343_T"
    rec_299_T = "rec_299_T"
    rec_172_T = "rec_172_T"
    rec_394_T = "rec_394_T"
    rec_218_T = "rec_218_T"
    rec_347_T = "rec_347_T"
    rec_295_T = "rec_295_T"
    rec_88_T = "rec_88_T"
    rec_239_T = "rec_239_T"
    rec_472_T = "rec_472_T"
    rec_235_T = "rec_235_T"
    rec_330_T = "rec_330_T"
    rec_315_T = "rec_315_T"
    rec_374_T = "rec_374_T"
    rec_215_T = "rec_215_T"
    rec_256_T = "rec_256_T"
    rec_491_T = "rec_491_T"
    rec_496_T = "rec_496_T"
    rec_271_T = "rec_271_T"
    rec_439_T = "rec_439_T"
    rec_322_T = "rec_322_T"
    rec_216_T = "rec_216_T"
    rec_168_T = "rec_168_T"
    rec_207_T = "rec_207_T"
    rec_264_T = "rec_264_T"
    rec_106_T = "rec_106_T"
    rec_471_T = "rec_471_T"
    rec_450_T = "rec_450_T"
    rec_122_T = "rec_122_T"
    rec_340_T = "rec_340_T"
    rec_376_T = "rec_376_T"
    rec_171_T = "rec_171_T"
    rec_333_T = "rec_333_T"
    rec_31_T = "rec_31_T"
    rec_326_T = "rec_326_T"
    rec_430_T = "rec_430_T"
    rec_336_T = "rec_336_T"
    rec_9_T = "rec_9_T"
    rec_414_T = "rec_414_T"
    rec_151_T = "rec_151_T"
    rec_428_T = "rec_428_T"
    rec_148_T = "rec_148_T"
    rec_75_T = "rec_75_T"
    rec_179_T = "rec_179_T"
    rec_377_T = "rec_377_T"
    rec_93_T = "rec_93_T"
    rec_342_T = "rec_342_T"
    rec_310_T = "rec_310_T"
    rec_124_T = "rec_124_T"
    rec_217_T = "rec_217_T"
    rec_385_T = "rec_385_T"
    rec_499_T = "rec_499_T"
    rec_147_T = "rec_147_T"
    rec_246_T = "rec_246_T"
    rec_175_T = "rec_175_T"
    rec_378_T = "rec_378_T"
    rec_480_T = "rec_480_T"
    rec_363_T = "rec_363_T"
    rec_307_T = "rec_307_T"
    rec_360_T = "rec_360_T"
    rec_252_T = "rec_252_T"
    rec_469_T = "rec_469_T"
    rec_160_T = "rec_160_T"
    rec_63_T = "rec_63_T"
    rec_445_T = "rec_445_T"
    rec_402_T = "rec_402_T"
    rec_53_T = "rec_53_T"
    rec_48_T = "rec_48_T"
    rec_59_T = "rec_59_T"
    rec_277_T = "rec_277_T"
    rec_281_T = "rec_281_T"
    rec_321_T = "rec_321_T"
    rec_429_T = "rec_429_T"
    rec_181_T = "rec_181_T"
    rec_488_T = "rec_488_T"
    rec_452_T = "rec_452_T"
    rec_89_T = "rec_89_T"
    rec_305_T = "rec_305_T"
    rec_320_T = "rec_320_T"
    rec_205_T = "rec_205_T"
    rec_449_T = "rec_449_T"
    rec_85_T = "rec_85_T"
    rec_242_T = "rec_242_T"
    rec_498_T = "rec_498_T"
    rec_221_T = "rec_221_T"
    rec_367_T = "rec_367_T"
    rec_466_T = "rec_466_T"
    rec_323_T = "rec_323_T"
    rec_423_T = "rec_423_T"
    rec_425_T = "rec_425_T"
    rec_99_T = "rec_99_T"
    rec_142_T = "rec_142_T"
    rec_16_T = "rec_16_T"
    rec_14_T = "rec_14_T"
    rec_133_T = "rec_133_T"
    rec_98_T = "rec_98_T"
    rec_339_T = "rec_339_T"
    rec_78_T = "rec_78_T"
    rec_399_T = "rec_399_T"
    rec_197_T = "rec_197_T"
    rec_20_T = "rec_20_T"
    rec_196_T = "rec_196_T"
    rec_353_T = "rec_353_T"
    rec_45_T = "rec_45_T"
    rec_114_T = "rec_114_T"
    rec_386_T = "rec_386_T"
    rec_127_T = "rec_127_T"
    rec_482_T = "rec_482_T"
    rec_278_T = "rec_278_T"
    rec_209_T = "rec_209_T"
    rec_17_T = "rec_17_T"
    rec_140_T = "rec_140_T"
    rec_199_T = "rec_199_T"
    rec_50_T = "rec_50_T"
    rec_231_T = "rec_231_T"
    rec_274_T = "rec_274_T"
    rec_372_T = "rec_372_T"
    rec_382_T = "rec_382_T"
    rec_435_T = "rec_435_T"
    rec_393_T = "rec_393_T"
    rec_4_T = "rec_4_T"
    rec_389_T = "rec_389_T"
    rec_296_T = "rec_296_T"
    rec_475_T = "rec_475_T"
    rec_486_T = "rec_486_T"
    rec_470_T = "rec_470_T"
    rec_60_T = "rec_60_T"
    rec_427_T = "rec_427_T"
    rec_400_T = "rec_400_T"
    rec_319_T = "rec_319_T"
    rec_303_T = "rec_303_T"
    rec_424_T = "rec_424_T"
    rec_101_T = "rec_101_T"
    rec_476_T = "rec_476_T"
    rec_200_T = "rec_200_T"
    rec_413_T = "rec_413_T"
    rec_451_T = "rec_451_T"
    rec_275_T = "rec_275_T"
    rec_301_T = "rec_301_T"
    rec_208_T = "rec_208_T"
    rec_233_T = "rec_233_T"
    rec_49_T = "rec_49_T"
    rec_415_T = "rec_415_T"
    rec_465_T = "rec_465_T"
    rec_187_T = "rec_187_T"
    rec_87_T = "rec_87_T"
    rec_137_T = "rec_137_T"
    rec_426_T = "rec_426_T"
    rec_462_T = "rec_462_T"
    rec_143_T = "rec_143_T"
    rec_177_T = "rec_177_T"
    rec_357_T = "rec_357_T"
    rec_364_T = "rec_364_T"
    rec_161_T = "rec_161_T"
    rec_293_T = "rec_293_T"
    rec_153_T = "rec_153_T"
    rec_254_T = "rec_254_T"
    rec_396_T = "rec_396_T"
    rec_46_T = "rec_46_T"
    rec_487_T = "rec_487_T"
    rec_80_T = "rec_80_T"
    rec_83_T = "rec_83_T"
    rec_463_T = "rec_463_T"
    rec_325_T = "rec_325_T"
    rec_51_T = "rec_51_T"
    rec_398_T = "rec_398_T"
    rec_258_T = "rec_258_T"
    rec_198_T = "rec_198_T"
    rec_308_T = "rec_308_T"
    rec_224_T = "rec_224_T"
    rec_74_T = "rec_74_T"
    rec_370_T = "rec_370_T"
    rec_26_T = "rec_26_T"
    rec_350_T = "rec_350_T"
    rec_361_T = "rec_361_T"
    rec_497_T = "rec_497_T"
    rec_103_T = "rec_103_T"
    rec_125_T = "rec_125_T"
    rec_102_T = "rec_102_T"
    rec_12_T = "rec_12_T"
    rec_359_T = "rec_359_T"
    rec_182_T = "rec_182_T"
    rec_95_T = "rec_95_T"
    rec_105_T = "rec_105_T"
    rec_237_T = "rec_237_T"
    rec_387_T = "rec_387_T"
    rec_210_T = "rec_210_T"
    rec_332_T = "rec_332_T"
    rec_111_T = "rec_111_T"
    rec_232_T = "rec_232_T"
    rec_495_T = "rec_495_T"
    rec_481_T = "rec_481_T"
    rec_483_T = "rec_483_T"
    rec_129_T = "rec_129_T"
    rec_354_T = "rec_354_T"
    rec_434_T = "rec_434_T"
    rec_300_T = "rec_300_T"
    rec_169_T = "rec_169_T"
    rec_117_T = "rec_117_T"
    rec_381_T = "rec_381_T"
    rec_126_T = "rec_126_T"
    rec_77_T = "rec_77_T"
    rec_334_T = "rec_334_T"
    rec_443_T = "rec_443_T"
    rec_380_T = "rec_380_T"
    rec_69_T = "rec_69_T"
    rec_145_T = "rec_145_T"
    rec_265_T = "rec_265_T"
    rec_493_T = "rec_493_T"
    rec_174_T = "rec_174_T"
    rec_10_T = "rec_10_T"
    rec_454_T = "rec_454_T"
    rec_213_T = "rec_213_T"
    rec_268_T = "rec_268_T"
    rec_309_T = "rec_309_T"
    rec_165_T = "rec_165_T"
    rec_478_T = "rec_478_T"
    rec_30_T = "rec_30_T"
    rec_494_T = "rec_494_T"
    rec_64_T = "rec_64_T"
    rec_276_T = "rec_276_T"
    rec_419_T = "rec_419_T"
    rec_66_T = "rec_66_T"
    rec_279_T = "rec_279_T"
    rec_188_T = "rec_188_T"
    rec_440_T = "rec_440_T"
    rec_191_T = "rec_191_T"
    rec_19_T = "rec_19_T"
    rec_194_T = "rec_194_T"
    rec_202_T = "rec_202_T"
    rec_214_T = "rec_214_T"
    rec_149_T = "rec_149_T"
    rec_178_T = "rec_178_T"
    rec_455_T = "rec_455_T"
    rec_92_T = "rec_92_T"
    rec_236_T = "rec_236_T"
    rec_241_T = "rec_241_T"
    rec_118_T = "rec_118_T"
    rec_116_T = "rec_116_T"
    rec_100_T = "rec_100_T"
    rec_263_T = "rec_263_T"
    rec_156_T = "rec_156_T"
    rec_5_T = "rec_5_T"
    rec_464_T = "rec_464_T"
    rec_219_T = "rec_219_T"
    rec_259_T = "rec_259_T"
    rec_176_T = "rec_176_T"
    rec_269_T = "rec_269_T"
    rec_211_T = "rec_211_T"
    rec_397_T = "rec_397_T"
    rec_352_T = "rec_352_T"
    rec_247_T = "rec_247_T"
    rec_29_T = "rec_29_T"
    rec_54_T = "rec_54_T"
    rec_27_T = "rec_27_T"
    rec_290_T = "rec_290_T"
    rec_28_T = "rec_28_T"
    rec_8_T = "rec_8_T"
    rec_167_T = "rec_167_T"
    rec_136_T = "rec_136_T"
    rec_302_T = "rec_302_T"
    rec_110_T = "rec_110_T"
    rec_155_T = "rec_155_T"
    rec_262_T = "rec_262_T"
    rec_76_T = "rec_76_T"
    rec_446_T = "rec_446_T"
    rec_73_T = "rec_73_T"
    rec_225_T = "rec_225_T"
    rec_62_T = "rec_62_T"
    rec_56_T = "rec_56_T"
    rec_109_T = "rec_109_T"
    rec_484_T = "rec_484_T"
    rec_331_T = "rec_331_T"
    rec_344_T = "rec_344_T"
    rec_468_T = "rec_468_T"
    rec_58_T = "rec_58_T"
    rec_163_T = "rec_163_T"
    rec_369_T = "rec_369_T"
    rec_162_T = "rec_162_T"
    rec_3_T = "rec_3_T"
    rec_91_T = "rec_91_T"
    rec_273_T = "rec_273_T"
    rec_115_T = "rec_115_T"
    rec_412_T = "rec_412_T"
    rec_418_T = "rec_418_T"
    rec_404_T = "rec_404_T"
    rec_68_T = "rec_68_T"
    rec_261_T = "rec_261_T"
    rec_292_T = "rec_292_T"
    rec_42_T = "rec_42_T"
    rec_135_T = "rec_135_T"
    rec_2_T = "rec_2_T"
    rec_318_T = "rec_318_T"
    rec_338_T = "rec_338_T"
    rec_395_T = "rec_395_T"
    rec_36_T = "rec_36_T"
    rec_71_T = "rec_71_T"
    rec_107_T = "rec_107_T"
    rec_227_T = "rec_227_T"
    rec_442_T = "rec_442_T"
    rec_39_T = "rec_39_T"
    rec_35_T = "rec_35_T"
    rec_436_T = "rec_436_T"
    rec_90_T = "rec_90_T"
    rec_130_T = "rec_130_T"
    rec_477_T = "rec_477_T"
    rec_195_T = "rec_195_T"
    rec_96_T = "rec_96_T"
    rec_467_T = "rec_467_T"
    rec_183_T = "rec_183_T"
    rec_351_T = "rec_351_T"
    rec_226_T = "rec_226_T"
    rec_312_T = "rec_312_T"
    rec_131_T = "rec_131_T"
    rec_433_T = "rec_433_T"
    rec_41_T = "rec_41_T"
    rec_298_T = "rec_298_T"
    rec_112_T = "rec_112_T"
    rec_132_T = "rec_132_T"
    rec_13_T = "rec_13_T"
    rec_65_T = "rec_65_T"
    rec_201_T = "rec_201_T"
    rec_61_T = "rec_61_T"
    rec_457_T = "rec_457_T"
    rec_238_T = "rec_238_T"
    rec_417_T = "rec_417_T"
    rec_138_T = "rec_138_T"
    rec_15_T = "rec_15_T"
    rec_348_T = "rec_348_T"
    rec_192_T = "rec_192_T"
    rec_337_T = "rec_337_T"
    rec_314_T = "rec_314_T"
    rec_335_T = "rec_335_T"
    rec_222_T = "rec_222_T"
    rec_356_T = "rec_356_T"
    rec_37_T = "rec_37_T"
    rec_250_T = "rec_250_T"
    rec_150_T = "rec_150_T"
    rec_329_T = "rec_329_T"
    rec_119_T = "rec_119_T"
    rec_294_T = "rec_294_T"
    rec_365_T = "rec_365_T"
    rec_154_T = "rec_154_T"
    rec_113_T = "rec_113_T"
    rec_128_T = "rec_128_T"
    rec_355_T = "rec_355_T"
    rec_388_T = "rec_388_T"
    rec_82_T = "rec_82_T"
    rec_79_T = "rec_79_T"
    rec_328_T = "rec_328_T"
    rec_459_T = "rec_459_T"
    rec_22_T = "rec_22_T"
    rec_282_T = "rec_282_T"
    rec_11_T = "rec_11_T"
    rec_373_T = "rec_373_T"
    rec_362_T = "rec_362_T"
    rec_244_T = "rec_244_T"
    rec_317_T = "rec_317_T"
    rec_311_T = "rec_311_T"
    rec_144_T = "rec_144_T"
    rec_146_T = "rec_146_T"
    rec_123_T = "rec_123_T"
    rec_72_T = "rec_72_T"
    rec_248_T = "rec_248_T"
    rec_47_T = "rec_47_T"
    rec_180_T = "rec_180_T"
    rec_341_T = "rec_341_T"
    rec_444_T = "rec_444_T"
    rec_375_T = "rec_375_T"
    rec_249_T = "rec_249_T"
    rec_406_T = "rec_406_T"
    rec_81_T = "rec_81_T"
    rec_104_T = "rec_104_T"
    rec_316_T = "rec_316_T"
    rec_55_T = "rec_55_T"
    rec_203_T = "rec_203_T"
    rec_304_T = "rec_304_T"
    rec_186_T = "rec_186_T"
    rec_379_T = "rec_379_T"
    rec_255_T = "rec_255_T"
    rec_206_T = "rec_206_T"
    rec_289_T = "rec_289_T"
    rec_24_T = "rec_24_T"
    rec_170_T = "rec_170_T"
    rec_448_T = "rec_448_T"
    rec_234_T = "rec_234_T"
    rec_437_T = "rec_437_T"
    rec_253_T = "rec_253_T"
    rec_283_T = "rec_283_T"
    rec_70_T = "rec_70_T"
    rec_313_T = "rec_313_T"
    rec_270_T = "rec_270_T"
    rec_7_T = "rec_7_T"
    rec_460_T = "rec_460_T"
    rec_324_T = "rec_324_T"
    rec_346_T = "rec_346_T"
    rec_97_T = "rec_97_T"
    rec_166_T = "rec_166_T"
    rec_190_T = "rec_190_T"
    rec_441_T = "rec_441_T"
    rec_139_T = "rec_139_T"
    rec_432_T = "rec_432_T"
    rec_52_T = "rec_52_T"
    rec_34_T = "rec_34_T"
    rec_366_T = "rec_366_T"
    rec_94_T = "rec_94_T"
    rec_23_T = "rec_23_T"
    rec_84_T = "rec_84_T"
    rec_490_T = "rec_490_T"
    rec_43_T = "rec_43_T"
    rec_288_T = "rec_288_T"
    rec_409_T = "rec_409_T"
    rec_422_T = "rec_422_T"
    rec_287_T = "rec_287_T"
    rec_121_T = "rec_121_T"
    rec_421_T = "rec_421_T"
    rec_403_T = "rec_403_T"
    rec_230_T = "rec_230_T"
    rec_453_T = "rec_453_T"
    rec_266_T = "rec_266_T"
    rec_420_T = "rec_420_T"
    rec_492_T = "rec_492_T"
    rec_164_T = "rec_164_T"
    rec_291_T = "rec_291_T"
    rec_272_T = "rec_272_T"
    rec_220_T = "rec_220_T"
    rec_345_T = "rec_345_T"
    rec_228_T = "rec_228_T"
    rec_485_T = "rec_485_T"
    rec_141_T = "rec_141_T"
    rec_368_T = "rec_368_T"
    rec_410_T = "rec_410_T"
    rec_297_T = "rec_297_T"
    rec_392_T = "rec_392_T"
    rec_86_T = "rec_86_T"
    rec_280_T = "rec_280_T"
    rec_391_T = "rec_391_T"
    rec_438_T = "rec_438_T"
    rec_6_T = "rec_6_T"
    rec_383_T = "rec_383_T"
    rec_286_T = "rec_286_T"
    rec_33_T = "rec_33_T"
    rec_251_T = "rec_251_T"
    rec_349_T = "rec_349_T"
    rec_184_T = "rec_184_T"
    rec_38_T = "rec_38_T"
    rec_416_T = "rec_416_T"
    rec_120_T = "rec_120_T"
    rec_474_T = "rec_474_T"
    rec_267_T = "rec_267_T"
    rec_456_T = "rec_456_T"
    rec_461_T = "rec_461_T"
    rec_458_T = "rec_458_T"
    rec_473_T = "rec_473_T"
    rec_173_T = "rec_173_T"
