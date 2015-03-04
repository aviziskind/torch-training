doNoisyLetters = function() -- (allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 
    print('Starting Noisy Letters Experiment')

    local logNoisySet = onLaptop and true
    local showDetailedOptions = true


    local doConvNet      = (modelName == 'ConvNet')
    local doTextureModel = (modelName == 'Texture')
    local doOverFeat     = (modelName == 'OverFeat')
    
    local channels_doSVHN = false
    --local channelTuningStage = 'train'
    local channelTuningStage = 'test'
    
    --local channelTuningTestOn = 'hiLo'
    local channelTuningTestOn = 'band'
    
    
    --local groupingStage = 'train'
    local groupingStage = 'test'
    
    --local grouping_trainOn = 'SVHN'
        local grouping_SVHN_settings = {size = {32, 32}, globalNorm = true, localContrastNorm = false}
    --local grouping_trainOn = 'allWiggles'
    local grouping_trainOn = 'noWiggle'
    --local grouping_trainOn = 'Sloan'
    --local grouping_trainOn = 'sameWiggle'

    local repeatUntilNoSkips = true
    
    local complexity_trainWithPinkNoise = true
    
    local complexityStage = 'train'
    --local complexityStage = 'test'
    
    local netType, stimType, allSNRs
    if doConvNet then
        netType = 'ConvNet'
        stimType = 'NoisyLetters'
    elseif doTextureModel then
        netType = 'MLP'
        stimType = 'NoisyLettersTextureStats'
    elseif doOverFeat then        
        netType = 'MLP'
        stimType = 'NoisyLettersOverFeat'
    end


    
    local nTrials = 1

    ----------------------------------------------------------------------------------------------------------
    -------------------------------------------- NETWORK OPTIONS  --------------------------------------------
    ----------------------------------------------------------------------------------------------------------

   
    if logNoisySet then
        io.write('Save Current script to log file? [Y/n]')
        local ans = io.read("*line")
        if ans == 'n' or ans == 'N' then
            logNoisySet = false
        end        
        
    end
   
    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.01  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 10
    trainOpts.MAX_EPOCHS = 200
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1

    
    
   
    local setName 
    setName = '' 
    --local setName = 'nStates'
    --setName = 'poolSizes'
    --setName = 'filtSizes' 
    --setName = 'filtSizes_poolSizes'
    --setName = 'LeNet'
    
    
    
    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
    
    local allNetworks, allNetNames, allNetNames_nice
    local allNetworkOptions
    local convFunction
    if onLaptop then
        if useGPU then
        --convFunction = 'SpatialConvolutionMap' 
            convFunction = 'SpatialConvolutionCUDA'
        else
            convFunction = 'SpatialConvolution'
        end
    else
        if useGPU then
            convFunction = 'SpatialConvolutionCUDA'
            --convFunction = 'SpatialConvolution'        
        else
            convFunction = 'SpatialConvolution'
        end

    end

    local snr_train012 = {0,1,2}
    local snr_train_n1t2 = {-1,0,1,2}
    local snr_train_n2t1 = {-2,-1,0,1}
    local snr_train1t3 = {1,2,3}
    local snr_train1t4 = {1,2,3,4}
    local snr_train1h3 = table.range(1, 3, 0.5) --{0,1,2}
    local snr_train1h3p5 = table.range(1, 3.5, 0.5) --{0,1,2}
    local snr_train4 = {4}
        
    local snr_train, snr_train2, tbl_allSNRs_train
    if netType == 'ConvNet' then
        -- ConvNet options

        
        if expName == 'ChannelTuning' then
            tbl_allSNRs_train = { snr_train1h3 }
            --tbl_allSNRs_train = { snr_train_n1t2, snr_train_n2t1 }
        elseif expName == 'Uncertainty' or expName == 'Complexity' or expName == 'Grouping' then
            tbl_allSNRs_train = { snr_train1h3 }
        end
        
        --tbl_allSNRs_train = { snr_train, snr_train2 }
        
        --snr_train = {4}

        --snr_train = {1,2,3,4}
        --snr_train = {0,1,2,3,4}


        --allNStates = { {6, -50}, {6, -100},  {12, -50}, {12, -100} }
        --allNStates = { {6, -100}  }
        --allNStates = {  {6, 16, -120}, {9,24, -180}, {12,32,-240} } -- {60, 160, -1200} }
        --allNStates = {  {6, 16, -120}, {12,32,-240}, {60, 160, -1200} } -- {60, 160, -1200} }
        
        --allNStates = { {3, 8, -60}, {6, 16, -120}, {6, 16, -30}, {12,32,-240}, } -- {60, 160, -1200} }
        --allNStates = {  {6, 16, -120}, } --  {12, 32, -120}, {24, 64, -120},  {48, 128, -120} } 
        --allNStates = {  {6, 16, -15},  {6, 16, -30}, {6, 16, -60},  {6, 16, -120}, {6, 16, -240}, {6, 16, -480}, {6, 16, -960} } -- {60, 160, -1200} }
        
        --allNStates = {  {6, 16, -120},  {6, 32, -120}, {6, 64, -120},  {6, 128, -120}, {6, 16, -960}  } -- {60, 160, -1200} }
        --allNStates = {  {3, 16, -120}, {6, 16, -120},  {12, 16, -120}, {24, 16,-120}, {48, 16, -120},   } -- {60, 160, 1200} }
        --allNStates = {  {3, 8, -60}, {6, 16, -120}, {6, 16, -30}, {6,16, -15}, {6,16,-8}, {12,32,-240},  {3,8,-10}, {3,8,-5}, {3,8,-3}, {3,8,-30}, {6,8,-10}, {12,8,-10} }
        --allNStates = {  {6, 16, -120}, {}
            
        --allNStates = { {6, 16, -960} ,
          --              {6, 4, -120},  {6, 8, -120},  {6, 16, -120},  {6, 32, -120}, {6, 64, -120},  {6, 128, -120}, {6, 256, -120},  -- {60, 160, -1200} }
            --             {3, 16, -120}, {6, 16, -120},  {12, 16, -120}, {24, 16, -120}, {48, 16, -120}, {96, 16, -120}, {192, 16, -120}  } -- {60, 160, -1200} }
            
        --local allNStates = { {6, -15}, {6, -30}, {6,-60}, {6,-120}, {6,-240}, {12, -15}, {12, -30}, {12,-60}, {12,-120}, {12, -240} }
        
        --local allNStates = { {12, -60}, {24, -120}, {48, -120} }
        
--        local allNStates = {  {16, 32, -120}, {16, -120}, }
        --local allNStates = {  {3, -15}, {6, -15}, {12, -15},  {24, -15},   {6, -30}, {6, -60}, {6, -120},   {16, -30}, {16, -60}, {16, -120}    }
        --local allNStates = {  {16, -120} }
        
        --local allNStates = { {6, 16, -120}, {16, 32, -120},  {6, 16, 120, -84},   }
        --local allNStates = { {6, -120}, }
        --local allNStates = {  {16, 32, -32}, }--{16, 128, -32}, {16, 512, -32}, {16, 32, -120},  {16, 128, -120},  }
        --local allNStates = { {16, 32, -120}, {16, 128, -120},  }
        
        --local allNStates = { {6, 16, -120}, {16, 32, -240}, {16, 32, -120}, {16, 128, -120},  }
        --local allNStates = { {6, 16, -120} }
        local allNStates = {  {24, 16, -240}  }
        --local allNStates = { {6, 16, -120}, {24, 16, -120}, {16, 64, -120},   }
        --local allNStates = { {6, 16, -120}, {16, 32, -120}, {16, 32, -240}, {16, 128, -120},  }
        
        
        if onLaptop then
            --allNStates = { {16, -120} }
        end
            
        --local allNStates = { {6,-15}, {6,-30} }
        
        --local allNStates = {12, -60}, {24,-120}, {48,-120}

        local filtSizes = {5, 4}
        --local allFiltSizes = { {5, 5, 5}}
        --local allFiltSizes = { {3, 3}, {5, 5}, }
        local allFiltSizes = {  {5, 5}, {7, 7} }
        --local allFiltSizes = { {5 } }-- {5, 7}  }
        --local allFiltSizes = {  5, 10, 20 }
        
        local doPooling = true
        --local allDoPooling = {false, true}
        local allDoPooling = {true}

        
        --local filtSizes = {34}
        
        local poolType = 'MAX'
        --local allPoolTypes = {2, 'MAX'}  -- {1, 2, 'MAX'}
        --local allPoolTypes = {2}  -- {1, 2, 'MAX'}
        local allPoolTypes = {2, 'MAX'}  -- {1, 2, 'MAX'}

        
        local poolStride = 'auto'
        --local poolStride = 2
        
        --local allPoolStrides = {'auto', 2,4,6,8}
        --local allPoolStrides = {'auto', 2}
        local allPoolStrides = {'auto'}
        --local allPoolStrides = {2,4}
            
        local poolSize = {4, 2}        
        --local allPoolSizes = { {4, 2}, {4, 4}, {2, 2}, {2, 4}, }
        --local allPoolSizes = { {4, 2}, {0, 0} } 
        --local allPoolSizes = { {4, 0, 0}, {2, 2, 0}, {0, 0, 0},  } 
        --local allPoolSizes = { 0, 2, 4, 6, 7, 8, 10, 12, 14} 
        --local allPoolSizes = {0, 2, 4, 6, 8, 12, 16, 20} 
        --local allPoolSizes = { {0, 0}, {2, 2}, {4, 0}, {4, 2} } 
        --local allPoolSizes = { 6 }
        --local allPoolSizes = { {4, 0}, {4, 2}, {4, 4}, {4, 6} } 
        --local allPoolSizes = {2, 3, 4, 6, 8}
        --local allPoolSizes = {2, 3, 4, 6, 8}
        --local allPoolSizes = {2, 4, 8, 0} 
        --local allPoolSizes = { {2, 2, 2}, {2, 2, 0} } 
        --local allPoolSizes = { {2, 2}, {4, 2}  } 
        
        --local allPoolSizes = { {2, 2}, {3, 2}, {4, 2} } 
        local allPoolSizes = { {2, 2}, {3, 2}, {4, 2} } 
                
        if onLaptop then
            --allPoolSizes = { 6 }
        end

        
        if setName == 'nStates' then
            --allNStates =  { {3,-15}, {6,-15}, {12,-15}, {24,-15},   {6,-8}, {6,-30}, {6,-60}  } 
            --allNStates =  { {6,-15}, {6,-60}, {6,-120}, {6,-240}  } 
            --allNStates =  { {6,-15},  {6,-120}, {6},  {6, 16},  {6, 16, -120},  } --{6, 16}, {6, 16,  }
            
            allNStates =  {   {6, -15}, {6, -120},   }
            
            --allNStates =  {   {6, 16},  {6, 16, -120},   {6,16, -240}, {12, 32},  {12, 32, -240}   }  -- {24, 64, -480}
             
            
        elseif setName == 'filtSizes' then
            --allFiltSizes = {2,3,4,5,10,20}
            allFiltSizes = {3,5,10}
            
        elseif setName == 'poolSizes' then
            --allDoPooling = {true}
            
            --[[
            allNStates = { { 1,15}, {1,60},  } --, {2, 30}  }
            allFiltSizes = {0}
            allPoolSizes = {0,2,4,6,8} --{0, 2,4,6,8}
            allPoolTypes = {1}
            --]]
            --allPoolSizes = {0,2,4,6,8} --{0, 2,4,6,8}
            allPoolSizes = {0, 2, 4, 6,8} --{0, 2,4,6,8}
            
        elseif setName == 'poolTypes' then
            allPoolTypes = {1, 2, 'MAX'}
            
        elseif setName == 'filtSizes_poolSizes' then
            --allFiltSizes = {5,10}
            --allPoolSizes = {0,2,4,6}

            --allFiltSizes = {2,3,5,8,16,20,24}
            --allPoolSizes = {0, 2,4,6,8,10,12}
            
            --allFiltSizes = {2,3,5,16,24}
            --allPoolSizes = {0, 2,4,6,8,12}

            --allFiltSizes = {3,5,10,20}
            --allPoolSizes = {0, 2,4,8,16}
            
            --allNStates =  {  {6}, {6, -15}, {6, -120}, {6, -480} }, -- {6, 16}, {6, 16, -30}, {6, 16, -120},  }            
            --allNStates =  {   {6, -120},  } -- {6, 16}, {6, 16, -30}, {6, 16, -120},  }            
            --allNStates =  {  {12, -60, -60} }            
            --allNStates =  {  {6, -120} }            
            --allNStates =  {  {24, -120}, }            
            allNStates =  {  {6, -15}, }            
            --allNStates =  {  {24}, {24, -15}, {24, -120}, {24, -480}, {96}, {96, -15}, {96, -120}, {96, -480},  }            
            --allNStates =  {  {20, 50, -500}  }            
            
            allFiltSizes = { {5, 4} }
            --allFiltSizes = { {5}, {10}, {20} }
            
            --allPoolSizes = { {0, 0}, {2, 2}, {4, 2} }
            allPoolSizes = { {0, 0}, {4, 2}  } 
            --allPoolSizes = { {2, 2} } 
        elseif setName == 'LeNet' then
            
            allNStates = { {6, 16, 120, -84}, }--{6,-15} }

            allFiltSizes = { {5, 5, 5} }
            
            --local allDoPooling = {false, true}
            allDoPooling = {true, false}
                        
            allPoolStrides = {'auto'}
               
            allPoolSizes = { {2, 2, 0} } 
           
           
        end
        
                
        
        local config_sgd = {learningRate = 1e-3,  learningRateDecay = 1e-4,  weightDecay = 0}
        local config_sgd_mom = {learningRate = 1e-3,  learningRateDecay = 1e-4, weightDecay = 0, momentum = 0.95, dampening = 0}
        local config_sgd_mom_nest = {learningRate = 1e-3,  learningRateDecay = 1e-4, weightDecay = 0, momentum = 0.95, dampening = 0, nesterov = 1}
        local config_adadelta = {adaptiveMethod = 'ADADELTA', rho = 0.95, epsilon = 1e-6}
        local config_rmsprop = {adaptiveMethod = 'rmsprop', rho = 0.95, epsilon = 1e-6}
        local config_vSGD = {adaptiveMethod = 'vSGD'}
            
        --local tbl_trainConfig = {config_vSGD, config_sgd, config_sgd_mom, config_adadelta, config_rmsprop, config_vSGD}
        --local tbl_trainConfig = {config_sgd, config_sgd_mom, config_adadelta}
        local tbl_trainConfig = {config_sgd_mom }
            
            
        

        allNetworkOptions = { netType = 'ConvNet', 
                                tbl_nStates = allNStates,
                                
                                --filtSizes = filtSizes,
                                tbl_filtSizes = allFiltSizes,
                                
                                --doPooling = doPooling, 
                                tbl_doPooling = allDoPooling,
                                
                                --poolSizes = poolSize,
                                tbl_poolSizes = allPoolSizes,
                                
                                --poolStrides = poolStride, 
                                tbl_poolStrides = allPoolStrides,
                                
                                --poolType = poolType,
                                tbl_poolType = allPoolTypes,
                                
                                --useConnectionTable = useConnectionTable,
                                convFunction = convFunction,
                                
                                trainOnGPU = useGPU,
                                
                                tbl_trainConfig = tbl_trainConfig,
                                
                              }
           
               
       
       --print('print1', allNetworks)    
        local allNetworks_orig = expandOptionsToList(allNetworkOptions)
        allNetworks = fixConvNetParams(allNetworks_orig)
        --print('print2', allNetworks)
        
        
    elseif netType == 'MLP' then
        --trainOpts.MIN_EPOCHS = 3
        
        --tbl_allSNRs_train = { snr_train1, snr_train2 }
        --tbl_allSNRs_train = { {3}, {4}, {3,4}, {2, 3, 4}, {2, 3, 4, 5} }
        --tbl_allSNRs_train = { {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}, {2, 3, 4}, {2, 3, 4, 5} }
        --tbl_allSNRs_train = { {2, 3, 4, 5} }
        tbl_allSNRs_train = { snr_train1t3, snr_train1t4 }
        
        if expName == 'ChannelTuning' then
            tbl_allSNRs_train = { snr_train1h3 }
            --tbl_allSNRs_train = { snr_train_n1t2, snr_train_n2t1 }
        elseif expName == 'Uncertainty' or expName == 'Complexity' or expName == 'Grouping' then
            tbl_allSNRs_train = { snr_train1h3 }
        end

        
--        local allNHiddenUnits = { {6}, {12}, {24}, {48}, {96},   {6, 16}, {6, 32}, {6, 64},   {12, 16}, {12, 32}, {12, 64} }
        --local allNHiddenUnits = { {}, {4}, {5}, {8}, {10}, {15}, {30}, {60}, {120} }
--        allNHiddenUnits = {4,10,20,40,100,200}
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}
        
        --local allNHiddenUnits = { {}, {30}, {120}, {240}, {480}, {60, 60} }
        --local allNHiddenUnits = { {}, {30}, {120}, {240}, }
        --local allNHiddenUnits = { {}, {120}, {240} }
        --local allNHiddenUnits = { {}, {120}, }
        local allNHiddenUnits = { {}, }
            
  --          {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        allNetworkOptions = { netType = 'MLP',
                              tbl_nHiddenUnits = allNHiddenUnits }                    
        
        allNetworks = expandOptionsToList(allNetworkOptions)
        
    end
    
    --logFile
    if logNoisySet then
        local date_time_str = string.gsub(string.gsub( os.date('%Y/%m/%d  %X', os.time()), '[ ]', '_' ), '[/:]', '-')
        
        local logFileName = torchLetters_dir .. 'log/logFile_' .. date_time_str .. '_' .. expName .. '.txt' 
        logFile = assert( io.open(logFileName, 'w') )
        io.write('[Writing to log file ...]\n')

    end
        
        
    
    
    allNetworks, allNetNames_nice, allNetNames = uniqueNetworks(allNetworks)        
    --print( '\n===Using the following networks options ===')
    print('\n===Using the following network options ===')
    print(optionsTable2string(allNetworkOptions))
    if showDetailedOptions then
        print('\n===Using the following networks ===')
        print(allNetNames)    
        print('\n===Using the following networks (full descriptions) ===')
        print(allNetNames_nice)
    end

    s = allNetNames
    S = allNetNames_nice
    if logNoisySet then
        logFile:write( table.tostring(string.format('\n\n\n ======================== %s ====================== \n', os.date('%x %X', os.time()))) ) 
        --logFile:write( '\n===Using the following networks options ===')
        --logFile:write( table.tostring(allNetworkOptions))
        logFile:write( '\n===Using the following network options ===\n')
        logFile:write( optionsTable2string(allNetworkOptions))
        
    end




    ----------------------------------------------------------------------------------------------------------
    -------------------------------------------- LETTER STIMULUS OPTIONS  ------------------------------------
    ----------------------------------------------------------------------------------------------------------

    ----------------
    ----- FONT NAMES

   
    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

   
    local styles_use = {'Roman', 'Bold', 'Italic', 'BoldItalic'}
    --local styles_use = {'Roman', 'Bold'}
    
    
    
    local allStdFontNames      = {'Bookman', 'Sloan', 'Helvetica', 'KuenstlerU', 'Braille', 'Yung'}
    local allStdFontNames_tbl  = { {'Bookman'}, {'Sloan'}, {'Helvetica'}, {'KuenstlerU'}, {'Braille'}, {'Yung'} }

    
    local Bookman_rBIJ = {fonts = {'Bookman'}, styles = styles_use};
    local mK_rBIJ = {fonts = {'Bookman', 'KuenstlerU'}, styles = styles_use};
    local mrh_rBIJ = {fonts = {'Bookman', 'Courier', 'Helvetica'}, styles = styles_use};
    --local mMlrhKsy_rBIJ = {fonts = allStdFontNames, styles = styles_use} 
    local mlhKsy = {fonts = allStdFontNames } 
    local mlhKsy_rBIJ = {fonts = allStdFontNames, styles = styles_use} 
    


        
    
    -- NOISE FILTERS

    local applyGainFactor_band = true
    local applyGainFactor_hiLo = false
    local applyGainFactor_pink = true
    
    --local allBandNoise_cycPerLet = {0.5, 0.8, 1.3, 2.0, 3.2, 5.1, 8.1, 13}
    --local allCycPerLet = {0.5, 0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8, 11.31, 16}
    local allCycPerLet = {0.5, 0.59, 0.71, 0.84, 1.00, 1.19, 1.41, 1.68, 2, 2.38, 2.83, 3.36, 4, 4.76, 5.66, 6.73, 8, 9.51, 11.31, 13.45, 16}
    --allCycPerLet = {1.3}
    --local allCycPerLet = {0.5}
    local tbl_allBandNoiseFilters = {}
    local tbl_allHiPassFilters = {}         
    local tbl_allLoPassFilters = {}
    for i,v in ipairs(allCycPerLet) do
        tbl_allBandNoiseFilters[i] = {filterType = 'band', cycPerLet_centFreq = v,   applyFourierMaskGainFactor = applyGainFactor_band}
        tbl_allHiPassFilters[i]    = {filterType = 'hi',   cycPerLet_cutOffFreq = v, applyFourierMaskGainFactor = applyGainFactor_hiLo}
        tbl_allLoPassFilters[i]    = {filterType = 'lo',   cycPerLet_cutOffFreq = v, applyFourierMaskGainFactor = applyGainFactor_hiLo}
    end
    local tbl_allBandHiLoNoiseFilters = table.merge(tbl_allBandNoiseFilters, tbl_allHiPassFilters, tbl_allLoPassFilters)
    local tbl_allHiLoNoiseFilters = table.merge(tbl_allHiPassFilters, tbl_allLoPassFilters)
    
    
    --local allPinkNoise_exp = {1, 1.5, 2}
    --local allPinkNoise_exp = {1, 1.7, 1.6, 1.5}
    local allPinkNoise_exp = {1, 1.6, 1.7}
    --allPinkNoise_exp = {1}
    --local allBandNoise_cycPerLet = {0.5}
    local tbl_allPinkNoiseFilters = {}
    local tbl_allPinkOrWhiteNoiseFilters = {}
    local tbl_allPinkPlusWhiteNoiseFilters = {}
    for i,v in ipairs(allPinkNoise_exp) do
        tbl_allPinkNoiseFilters[i]          = {filterType = '1/f',       f_exp = v, applyFourierMaskGainFactor = applyGainFactor_pink}
        tbl_allPinkOrWhiteNoiseFilters[i]   = {filterType = '1/fOwhite', f_exp = v, applyFourierMaskGainFactor = applyGainFactor_pink}
        tbl_allPinkPlusWhiteNoiseFilters[i] = {filterType = '1/fPwhite', f_exp = v, applyFourierMaskGainFactor = applyGainFactor_pink}
    end
    
    local whiteNoiseFilter = {filterType = 'white'}
    
    --local tbl_allPinkNoises = table.merge(tbl_allPinkNoiseFilters, tbl_allPinkOrWhiteNoiseFilters, tbl_allPinkPlusWhiteNoiseFilters, whiteNoiseFilter)
    --local tbl_allPinkNoises = table.merge(tbl_allPinkNoiseFilters, tbl_allPinkOrWhiteNoiseFilters, whiteNoiseFilter)
    local tbl_whiteNoiseAndPinkNoises = table.merge(tbl_allPinkNoiseFilters, whiteNoiseFilter)
    --tbl_allPinkNoises = table.merge(tbl_allPinkNoiseFilters)
    --local tbl_allPinkNoises = {whiteNoiseFilter}
    
    --local tbl_wiggleSettings = {}
    local tbl_trainingWiggle
    
    local allBlurs = {0}
    
    -----------------
    ---- UNCERTAINTY 
    local allOriXY_mult = {  {Nori = 1, dOri = 0,    Nx = 1, dX = 0,    Ny = 1, dY = 0}, 
                             --{Nori = 1, dOri = 0,    Nx = 2, dX = 4,    Ny = 1, dY = 0}, -- 2x1 [4] --> span = 4
                             --{Nori = 1, dOri = 0,    Nx = 3, dX = 2,    Ny = 1, dY = 0}, -- 3x1 [2] --> span = 4
                             --{Nori = 1, dOri = 0,    Nx = 5, dX = 1,    Ny = 1, dY = 0}, -- 5x1 [1] --> span = 4 
                             {Nori = 1, dOri = 0,    Nx = 3, dX = 4,    Ny = 1, dY = 0}, -- 3x1 [4] --> span = 8
                             {Nori = 1, dOri = 0,    Nx = 4, dX = 4,    Ny = 1, dY = 0},   -- 4x1 [4] --> span = 12
                             {Nori = 1, dOri = 0,    Nx = 4, dX = 4,    Ny = 4, dY = 4},   -- 4x4 [4] --> span = 12 x 12
                             {Nori = 1, dOri = 0,    Nx = 6, dX = 4,    Ny = 6, dY = 4},   -- 6x6 [4] --> span = 20 x 20
                             {Nori = 1, dOri = 0,    Nx = 11, dX = 2,    Ny = 11, dY = 2}, -- 11x11 [2] --> span = 20 x 20
                          }
        
    local allOriXY_one = {Nori = 1, dOri = 0,    Nx = 1, dX = 0,    Ny = 1, dY = 0}
    local allOriXY_3x3y = {Nori = 1, dOri = 0,    Nx = 3, dX = 1,    Ny = 3, dY = 1}
    local allOriXY_5x5y = {Nori = 1, dOri = 0,    Nx = 5, dX = 1,    Ny = 5, dY = 1}
    local allOriXY_9x9y7o = {Nori = 7, dOri = 5,    Nx = 9, dX = 3,    Ny = 9, dY = 3}
    local allOriXY_10x10y11o = {Nori = 11, dOri = 4,    Nx = 10, dX = 2,    Ny = 10, dY = 2}
    local allOriXY_10x10y21o = {Nori = 21, dOri = 2,    Nx = 10, dX = 2,    Ny = 10, dY = 2}
    local allOriXY_30x30y21o = {Nori = 21, dOri = 2,    Nx = 30, dX = 1,    Ny = 30, dY = 1}
    local allOriXY_19x19y21o = {Nori = 21, dOri = 2,    Nx = 19, dX = 1,    Ny = 19, dY = 1}
    local allOriXY_6x5y21o = {Nori = 21, dOri = 2,    Nx = 6, dX = 2,    Ny = 5, dY = 2}

    
    local allOriXY_scan = expandOptionsToList( {Nori = 1, dOri = 0,    Ny = 1, dY = 0, 
                                                tbl_Nx_and_dX =  {  --{2, 25}
                                                     --{2, 25}, {3, 25}, {4, 25} 
                                                       --{1, 0}, {2, 25}, {3, 25}, {4, 25} 
                                                      --{1, 0}, {51, 1}, {26, 2}, {17, 3}, {13, 4}, {11, 5}, {9, 6}, {7, 8}, {6, 10}, {4, 14}, {3, 18}, {2, 22},
                                                      --{9, 6}
                                                      {1, 0}, {51, 1}, {26, 2}, {17, 3}, {13, 4}, {11, 5}, {9, 6}, {6, 10}, {3, 18}, {2, 22},
                                                      --  {1, 0}, 
                                                        --{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 8}, {2, 16}, {2, 24},   
                                                        --{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 8}, {4, 16}, {4, 24}, 
                                                        --{9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 8}, {9, 16}, 
                                                     --{ {1, 0}, {51, 1},  {17, 3},  {11, 5},   {7, 8},  {4, 14},   {2, 22},
                                                     --{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 8}, {2, 10}, {2, 14}, {2, 18}, {2, 22},
                                                    }
                                                } );
            
    local allOriXY_large = {  {Nori = 11, dOri = 2,    Nx = 6, dX = 1,    Ny = 12, dY = 1}, 
                                {Nori = 1, dOri = 0,    Nx = 6, dX = 1,    Ny = 12, dY = 1}, 
                                {Nori = 11, dOri = 2,    Nx = 1, dX = 0,    Ny = 1, dY = 0},
                                 {Nori = 1, dOri = 0,    Nx = 1, dX = 0,    Ny = 1, dY = 0},
                            }
    
    local allOriXY_large_48 = {Nori = 11, dOri = 2,    Nx = 5, dX = 2,    Ny = 9, dY = 2}
    local allOriXY_large_48_k14 = {Nori = 11, dOri = 2,    Nx = 6, dX = 2,    Ny = 9, dY = 2}
                            
    local allOriXY_med1 = {  {Nori = 1, dOri = 0,    Nx = 6, dX = 1,    Ny = 12, dY = 1} }
    local allOriXY_test = {  {Nori = 3, dOri = 2,    Nx = 2, dX = 4,    Ny = 2, dY = 4} }
    
    local allOriXY_long = {  {Nori = 11, dOri = 2,    Nx = 24, dX = 2,    Ny = 5, dY = 1} }
    local allOriXY_long_32_128_dx2_ori = {Nori = 11, dOri = 2,    Nx = 49, dX = 2,    Ny = 1, dY = 0}
    
    local allOriXY_long_32_128_dx5      = {Nori = 1,  dOri = 0,    Nx = 20, dX = 5,    Ny = 1, dY = 0}
    local allOriXY_long_32_128_dx5_ori  = {Nori = 11, dOri = 2,    Nx = 20, dX = 5,    Ny = 1, dY = 0}
    local allOriXY_long_32_128_dx10     = {Nori = 1,  dOri = 0,    Nx = 10, dX = 10,   Ny = 1, dY = 0}
    local allOriXY_long_32_128_dx10_ori = {Nori = 11, dOri = 2,    Nx = 10, dX = 10,   Ny = 1, dY = 0}
    local allOriXY_6x9y21o = {Nori = 21, dOri = 2,    Nx = 6, dX = 2,   Ny = 9, dY = 2}
    
    local allOriXY_4x4y7o  = {Nori = 7,  dOri = 5,    Nx = 4, dX = 3,   Ny = 4, dY = 3}
    local allOriXY_6x6y11o = {Nori = 11, dOri = 3,    Nx = 6, dX = 2,   Ny = 6, dY = 2}
        
    
    local tbl_fontNames, allSNRs_test, tbl_OriXY, tbl_imageSize, tbl_sizeStyle, tbl_imageSize_and_sizeStyle
    local tbl_noiseFilter, tbl_trainingNoise, tbl_trainingFonts, tbl_retrainFromLayer, tbl_classifierForEachFont, tbl_trainingOriXY
    local tbl_trainingImageSize
    local loopKeysOrder 
    
    
    if expName == 'ChannelTuning' then
        print('Channel Tuning Experiment')
        tbl_fontNames = { {'Bookman'}  }
    
        
        --allSNRs_test = {-1, 0, 1, 2, 3, 4, 5, 6};
       
        
        
        tbl_OriXY = { allOriXY_4x4y7o , allOriXY_6x6y11o, allOriXY_one } --  allOriXY_6x5y21o, 
        --tbl_imageSize = { {64, 64}, {32, 32} }
        --tbl_sizeStyle = {'k40'}    
        --tbl_sizeStyle = {'k32'}    
        
        --tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k32'} , {{32, 32}, 'k16'},   }
        tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k32'}  }
         
        if channelTuningStage == 'train' then
            
            if channels_doSVHN then
                tbl_fontNames = { {'SVHN'}  }              --TRAIN ON SVHN
                 allSNRs_test = {0}
                
            else
                tbl_noiseFilter = tbl_whiteNoiseAndPinkNoises  -- TRAIN ON PINK (and/or WHITE) NOISES
                tbl_trainingNoise = {'same'}
                 allSNRs_test = table.range(-1, 5,  0.5);
            end
            
        elseif channelTuningStage == 'test' then
            
            if channelTuningTestOn == 'hiLo' then
                tbl_noiseFilter = tbl_allHiLoNoiseFilters    -- TEST ON BAND/HI/LO NOISE (after being trained on PINK NOISE or SVHN)
                 allSNRs_test = table.range(-3, 5,  0.5);
            
            elseif channelTuningTestOn == 'band' then
                tbl_noiseFilter = tbl_allBandNoiseFilters    -- TEST ON BAND/HI/LO NOISE (after being trained on PINK NOISE or SVHN)    
                 allSNRs_test = table.range(-1, 5,  0.5);
            end
            
            if channels_doSVHN then 
                tbl_trainingFonts = {'SVHN'}                 
                tbl_retrainFromLayer = {'classifier', 'linear'}
            else
                tbl_trainingNoise = tbl_whiteNoiseAndPinkNoises 
                
                tbl_retrainFromLayer = {'', 'classifier', 'linear'}
                --tbl_retrainFromLayer = {'classifier', ''}
                --tbl_retrainFromLayer = {''}
            end            
            loopKeysOrder  = {'noiseFilter', 'trainingNoise', 'OriXY', 'fontName', 'retrainFromLayer'};
        end 

        

    elseif expName == 'Uncertainty' then
        print('Uncertainty Experiment')
        tbl_fontNames = { Bookman_rBIJ, mrh_rBIJ, {'Bookman'},  }
        
        tbl_fontNames = table.merge( { allStdFontNames, 
                                   --{fonts = {'Bookman'}, styles = styles_use},  
                                   --{fonts = {'KuenstlerU'}, styles = styles_use}, 
                                   --{fonts = {'Braille'}, styles = styles_use} 
                                 } )
    
        tbl_fontNames = expandOptionsToList( { tbl_fonts = allStdFontNames_tbl,       
                                               tbl_styles = {styles_use, {'Roman'}} } ) 
        
        allSNRs_test = {0, 1, 2, 3, 4}
        
        tbl_OriXY = { allOriXY_long_32_128_dx5, allOriXY_long_32_128_dx5_ori, allOriXY_long_32_128_dx10, allOriXY_long_32_128_dx10_ori }
        
          --tbl_OriXY = { allOriXY_one }, 
          --tbl_OriXY = { allOriXY_one, allOriXY_large_48_k14}, 
          --tbl_OriXY  = { allOriXY_long_32_128_dx2_ori }, 
          --tbl_OriXY = { allOriXY_long_32_128_dx5, allOriXY_long_32_128_dx5_ori, allOriXY_long_32_128_dx10, allOriXY_long_32_128_dx10_ori }, 
        tbl_imageSize = { {32, 128}  }
        tbl_sizeStyle = {'k16'}
    
    elseif expName == 'Complexity' then
        
        print('Efficiency Vs Complexity Experiment')
        --tbl_fontNames = { {'Bookman', 'Courier'}  }
        
        tbl_fontNames = allStdFontNames
        --tbl_fontNames = {allStdFontNames}
        --tbl_fontNames = table.merge(allStdFontNames, {allStdFontNames})
                        
            
        if complexity_trainWithPinkNoise then
            if complexityStage == 'train' then
                tbl_noiseFilter = tbl_whiteNoiseAndPinkNoises 
                tbl_trainingNoise = {'same'}
                
            elseif complexityStage == 'test' then
                tbl_trainingNoise = tbl_whiteNoiseAndPinkNoises 
                tbl_noiseFilter = {whiteNoiseFilter}
                tbl_retrainFromLayer = {'', 'classifier', 'linear'}
                
                loopKeysOrder  = {'fontName', 'trainingNoise', 'OriXY', 'retrainFromLayer'};
            end
            
        end
        
            
        --tbl_fontNames = table.merge( { allStdFontNames, {fonts=allStdFontNames}, } )
        --tbl_fontNames = table.merge( { allStdFontNames, {fonts=allStdFontNames} , {fonts=allStdFontNames, styles= styles_use} } )
        
        --allSNRs_test = {0, 1, 2, 2.5, 3, 4}
        --allSNRs_test = {0, 0.5, 1, 1.5, 2, 2.5, 3, 4}
        allSNRs_test = table.range(0, 5,  0.5);
        
        --tbl_OriXY = { allOriXY_6x9y21o, allOriXY_one }
        tbl_OriXY = { allOriXY_4x4y7o , allOriXY_6x6y11o, allOriXY_6x9y21o, }
                
        --tbl_imageSize = {64, 64}          
        --tbl_sizeStyle = {'k16'}
        --tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k32'} , {{32, 32}, 'k16'},   }
        if doTextureModel then
            --tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k32'} }
        end
        --tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k32'} }
        tbl_imageSize_and_sizeStyle =  {  {{64, 64}, 'k24'} }
        tbl_classifierForEachFont = {false}
        --allBlurs = {0, 1, 1.5, 2}
        --allBlurs = {0, 1, 2}
        allBlurs = {0}
        
        

        
    elseif expName == 'Grouping' then
        
        print('Grouping Experiment')
                
                       
        local sloanTrainFonts = {{'Sloan'}, {'SloanO2'}, {'SloanT3'}}
        allSNRs_test = table.range(-1,  5,   0.5)
        
        if grouping_trainOn == 'same' then
            groupingStage = 'train'
        end
        
        local tbl_OriXY_forTraining
        
        tbl_OriXY_forTraining = { allOriXY_one, allOriXY_3x3y, allOriXY_5x5y }
        --tbl_OriXY = { allOriXY_9x9y7o }
        --tbl_OriXY_forTraining = { allOriXY_10x10y11o }
        --tbl_OriXY_forTraining = { allOriXY_10x10y21o }
        --tbl_OriXY_forTraining = { allOriXY_one }
        tbl_OriXY_forTraining = { allOriXY_30x30y21o }
        --tbl_OriXY_forTraining = { allOriXY_19x19y21o }
        
                

        
        
        --tbl_imageSize = { {32, 32}, {64, 64}  }
        --tbl_imageSize = { {32, 32},  }
        tbl_imageSize = { {96, 96}  }
        --tbl_imageSize = { {64, 64}  }
        
        --tbl_sizeStyle = { 55 }
        --tbl_sizeStyle = { 'k32' } 
        tbl_sizeStyle = { 'k48' } 
        
        
        local doNoWiggle = true
        local doOriWiggle    = true
        local doOffsetWiggle = true
        local doPhaseWiggle  = true
        --local wiggleType = 'offset'
        --local wiggleType = 'phase'
        
        local doMult = false and (sys.memoryAvailable()/1024 > 40)
        
        local oriAngles, offsetAngles, phaseAngles
        if doMult then
            oriAngles    = table.merge( {table.range(10, 90, 10), {table.range(0, 60, 10)} } )
            offsetAngles = table.merge( {table.range(10, 60, 10), {table.range(0, 60, 10)} } )
            phaseAngles = table.merge( {{1}, {{0, 1}}} )
        else
            oriAngles   = table.range(5, 90, 5)
            offsetAngles = table.range(5, 60, 5)
            phaseAngles = table.range(10, 10, 10)
        end
        
        
        
        local wiggleSettings_none, wiggleSettings_ori, wiggleSettings_offset, wiggleSettings_phase = {}, {}, {}, {}
        if doNoWiggle then
            wiggleSettings_none    = { none = 1 }
        end
        if doOriWiggle then
            wiggleSettings_ori     = expandOptionsToList( { tbl_orientation = oriAngles } )
        end
        if doOffsetWiggle then
            wiggleSettings_offset  = expandOptionsToList( { tbl_offset = offsetAngles } )
        end
        if doPhaseWiggle then
            wiggleSettings_phase   = expandOptionsToList( { tbl_phase = phaseAngles } )
        end
        local tbl_wiggleSettings = table.merge(wiggleSettings_none, wiggleSettings_ori, wiggleSettings_offset, wiggleSettings_phase)
                    
        
        
        local tbl_trainingWiggle_exp
        
        --tbl_wiggleSettings  = expandOptionsToList( { tbl_orientation = table.range(0, 30, 10) } )
        if grouping_trainOn == 'allWiggles' then
            assert(wiggleType)
            if wiggleType == 'orientation' then  
                tbl_trainingWiggle_exp  = { 'same', { orientation = oriAngles } }
            elseif wiggleType == 'offset' then   
                tbl_trainingWiggle_exp  = { 'same', { offset = offsetAngles } }
            elseif wiggleType == 'phase' then    
                tbl_trainingWiggle_exp  = { 'same', { phase = phaseAngles } }
            end
        elseif grouping_trainOn == 'noWiggle' then
            tbl_trainingWiggle_exp  = { wiggleSettings_none }
        else
            tbl_trainingWiggle_exp = {'same'}
        end
        
        
        
        if groupingStage == 'train' then
        
            if grouping_trainOn == 'SVHN' then
                tbl_fontNames = { {fonts = 'SVHN', svhn_opts = grouping_SVHN_settings} }
                tbl_imageSize = { grouping_SVHN_settings.size }
                tbl_OriXY         = { allOriXY_one }
                
            else
                
                if grouping_trainOn == 'Sloan' then
                    tbl_fontNames = sloanTrainFonts  
                
                elseif grouping_trainOn == 'allWiggles' or grouping_trainOn == 'noWiggle' or grouping_trainOn == 'sameWiggle' then
                    tbl_fontNames = expandOptionsToList(  { fonts = {'Snakes'}, tbl_wiggles = tbl_trainingWiggle_exp  } )
                    
                end
                
                tbl_noiseFilter = tbl_whiteNoiseAndPinkNoises  -- TRAIN ON PINK (and/or WHITE) NOISES
                tbl_trainingNoise = {'same'} 
                tbl_OriXY = tbl_OriXY_forTraining
                
                --tbl_wiggleSettings = trainingWiggle -- train on training wiggle
                
            end
            
        elseif groupingStage == 'test' then
            
            tbl_fontNames = expandOptionsToList(  { fonts = {'Snakes'}, tbl_wiggles = tbl_wiggleSettings  } )
            
            if grouping_trainOn == 'SVHN' then
                tbl_trainingFonts = { {fonts = 'SVHN', svhn_opts = grouping_SVHN_settings} }
                tbl_trainingImageSize = { {32, 32} }
                tbl_OriXY         = { allOriXY_one }
            else                
                
                if grouping_trainOn == 'Sloan' then
                    tbl_trainingFonts = sloanTrainFonts
                    
                elseif grouping_trainOn == 'allWiggles' or grouping_trainOn == 'noWiggle' then
                    tbl_trainingWiggle = tbl_trainingWiggle_exp
                    
                end
                
                tbl_trainingNoise = tbl_whiteNoiseAndPinkNoises     
            
                --tbl_OriXY         = { allOriXY_one, allOriXY_10x10y21o }
                tbl_OriXY         = tbl_OriXY_forTraining -- { allOriXY_19x19y21o }
                tbl_trainingOriXY = tbl_OriXY_forTraining
                
                --tbl_OriXY         = { allOriXY_one}
            end
            tbl_noiseFilter = { whiteNoiseFilter }
            
            tbl_retrainFromLayer = {'classifier', '', 'linear'}
            --tbl_retrainFromLayer = {'classifier', 'linear'}
            --tbl_retrainFromLayer = {'linear'}
            --tbl_retrainFromLayer = {''}
            
            --tbl_OriXY         = { allOriXY_one }
            
            
            loopKeysOrder  = {'trainingNoise', 'OriXY', 'fontName', 'retrainFromLayer'};
            
            
        end 
        
        
        
        
        
        
        
    elseif  expName == 'TrainingWithNoise' then
        print('Training With Noise Experiment')
        tbl_fontNames = { {'Bookman'}, {'KuenstlerU' } }
        tbl_imageSize = { {32, 32} }
        tbl_sizeStyle = { 'k16' }
        tbl_OriXY = { allOriXY_one }
        
        tbl_allSNRs_train = { {0}, {1}, {2}, {3}, {4},   {0, 1}, {1, 2}, {2, 3}, {3, 4},    {0, 1, 2}, {1, 2, 3}, {2, 3, 4},   
                                {0, 1, 2}, {.5, 1.5, 2.5}, {1, 2, 3}, {1.5, 2.5, 3.5}, {2, 3, 4}, 
                                {1, 1.5, 2, 2.5},  {1.5, 2, 2.5, 3}, {2, 2.5, 3, 3.5},
                                {1, 1.5, 2, 2.5, 3},  {1, 1.5, 2, 2.5, 3, 3.5}, {1, 2, 3, 4},   }
        --tbl_allSNRs_train = { {1}, {2}, {3}, {4}, }

        allSNRs_test = table.range(0, 4,   0.5)
    elseif expName == 'TestConvNet' then
        
        local oriXYSet_8x8y21o = {Nori = 21,  dOri = 2,    Nx = 8, dX = 2,    Ny = 8, dY = 2}
        
        tbl_fontNames = { {'Bookman'} }
        tbl_imageSize = { {40, 40} }
        tbl_sizeStyle = { 'k14' }
        --tbl_OriXY = { oriXYSet_8x8y21o }
        tbl_OriXY = { allOriXY_one }
        tbl_allSNRs_train = { {1, 2, 3}, }
            
        allSNRs_test = table.range(0, 4,   1)
        
    else
        error(string.format('Unknown experiment name: %s', expName))
        
    end
    
        
    
      
    local tbl_Nscl_txt, tbl_Nori_txt, tbl_Na_txt
    if doTextureModel then
        tbl_Nscl_txt = {4}
        tbl_Nori_txt = {4}
        tbl_Na_txt = {7}
       
    end


    local all_layerId, all_OF_contrast, all_OF_offset
    if doOverFeat then
        
              --local layerId = 19
        --local OF_contrast = 32
        --local OF_offset = 0
         
        all_layerId = {19, 17, 16}
        all_OF_contrast = {127, 64, 32, 16, 2}
        all_OF_offset = {0, 64, 127}
        
        --tbl_sizeStyle = {'k68'}
        tbl_sizeStyle = {'128-64'}
        --tbl_sizeStyle = {'128-64', 'k68'}
        tbl_imageSize = {231, 231}
    end
    
    
  
    
    
    
    local all_NoisyLetterOpts_tbl = { expName = expName, 
                                      stimType = stimType,
        
                                      tbl_fontName = tbl_fontNames,

                                      tbl_OriXY = tbl_OriXY,                                      
                                      
                                      autoImageSize = false,                                      
                                      
                                      tbl_imageSize = tbl_imageSize,
                                      tbl_sizeStyle = tbl_sizeStyle,
                                      tbl_imageSize_and_sizeStyle = tbl_imageSize_and_sizeStyle,                                      
                                      
                                      tbl_blurStd = allBlurs,
                                                                            
                                      tbl_noiseFilter = tbl_noiseFilter, tbl_trainingNoise = tbl_trainingNoise, tbl_retrainFromLayer = tbl_retrainFromLayer, 
                                      tbl_trainingFonts = tbl_trainingFonts,
									  
                                      doTextureStatistics = doTextureModel,  tbl_Nscl_txt = tbl_Nscl_txt, tbl_Nori_txt = tbl_Nori_txt, tbl_Na_txt = tbl_Na_txt, textureStatsUse = 'V2',
                                      
                                      doOverFeat = doOverFeat, networkId = 0,  tbl_layerId = all_layerId, tbl_OF_contrast = all_OF_contrast, tbl_OF_offset = all_OF_offset, 
									  
                                      tbl_trainOnIndividualPositions = {false},
                                      retrainOnCombinedPositions = false, 
                                    
                                      tbl_classifierForEachFont = tbl_classifierForEachFont,
                                      
                                      tbl_trainingWiggle = tbl_trainingWiggle, 
                                      tbl_trainingOriXY = tbl_trainingOriXY,
                                      tbl_trainingImageSize = tbl_trainingImageSize,
                                      
                                      tbl_SNR_train = tbl_allSNRs_train,
                                    }
    
    local all_NoisyLetterOpts = expandOptionsToList(all_NoisyLetterOpts_tbl, loopKeysOrder)
    NoisyLetterOpts = all_NoisyLetterOpts;
    --print(all_NoisyLetterOpts_tbl)
    
    for i,opt in ipairs(all_NoisyLetterOpts) do
        -- unpack values in OriXY to main struct.
        if (opt.OriXY) then   
            for k,v in pairs(opt.OriXY) do
                opt[k] = v
            end
        end     
        
        -- make sure fontName is in a table, even if just a single font.
        if type(opt.fontName) == 'string' then
            opt.fontName = {opt.fontName}
        end
        if type(opt.trainingFonts) == 'string' then
            opt.trainingFonts = {opt.trainingFonts}
        end
        
        -- remove 'retrainFromLayer' if trainNoise is same as testNoise, and trainFonts = testFonts
        local differentTrainTestNoise = opt.trainingNoise and (opt.trainingNoise ~= 'same') and (filterStr(opt.noiseFilter, 1) ~= filterStr(opt.trainingNoise, 1))
        local differentTrainTestFonts = opt.trainingFonts and (opt.trainingFonts ~= 'same') and (abbrevFontStyleNames(opt.trainingFonts) ~=  abbrevFontStyleNames(opt.fontName) )
                
        if not differentTrainTestNoise and not differentTrainTestFonts then
            opt.retrainFromLayer = nil
        end
        
    end
    
    L = all_NoisyLetterOpts
    local noisyLetterOpts_strs
    all_NoisyLetterOpts, noisyLetterOpts_strs = uniqueOpts(all_NoisyLetterOpts)
                    --[[
                all_NoisyLetterOpts = { { stimType = 'NoisyLetters', 
                                       Nori = 1, dOri = 0,    Nx = 1, dX = 0,    Ny = 1, dY = 0, 
                                       sizeStyle = 12,
                                       autoImageSize = false, 
                                       imageSize = {20, 20}, 
                                       SNR_train = snr_train
                                      } }
                              nTrials = 1
                --]]
                
                    
    print('\n===Using the following fonts : ===');
    
    local fontNames_use_strs = {}
    assert(#tbl_fontNames > 0)
    for i,fn_i in ipairs(tbl_fontNames) do
        fontNames_use_strs[i] = abbrevFontStyleNames(fn_i)
    end
    print(fontNames_use_strs)
    
    
    --[[
    local noisyLetterOpts_strs = {}
    for i,opt in ipairs(all_NoisyLetterOpts) do
        noisyLetterOpts_strs[i] = getLetterOptsStr(opt) 
    end
    --]]
    print('\n=== Using the following noisy-letter noisy-letter options : ===');
    print(optionsTable2string(all_NoisyLetterOpts_tbl))
    
    do
        --return
    end
    if showDetailedOptions then
        print('\n=== Using the following noisy-letter settings : ===');
        --table.sort(noisyLetterOpts_strs)
        print(noisyLetterOpts_strs)
    end
    
    
    
    if logNoisySet then
        
        logFile:write('\n\n')
        logFile:write('\n=== Using the following noisy-letter options : ===\n');
        logFile:write( optionsTable2string(all_NoisyLetterOpts_tbl))
        
        if showDetailedOptions or true then
            logFile:write( '\n\n\n\n More Details: \n ===Using the following networks ===')
            logFile:write( table.tostring(allNetNames))
            
            logFile:write(  '\n===Using the following networks (full descriptions) ===')
            logFile:write( table.tostring(allNetNames_nice))
            
            logFile:write('\n===Using the following fonts : ===\n ') 
            logFile:write(table.tostring( fontNames_use_strs))
        
            logFile:write('\n=== Using the following noisy-letter settings : ===\n');
            logFile:write(table.tostring(noisyLetterOpts_strs))  
            logFile:write('\n\n\n\n')
        end        

        logFile:close()
        io.write('[Finished writing to log file ...]\n')
        return
    end



    
                      --  {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  autoImageSize = false, imageSize = {45, 45}, noiseFilter = {cycPerLet_cent = 3.2}, trainOnWhiteNoise = true } }
        
    --[[
    local all_OriXY =  { {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 0,  autoImageSize = false, imageSize = {50,50} },
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 1,  autoImageSize = false, imageSize = {50,50} },
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 1.5,  autoImageSize = false, imageSize = {50,50}},
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 2,  autoImageSize = false, imageSize = {50,50}},
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 3,  autoImageSize = false, imageSize = {50,50}},
        --]]                 
        
                        -- imageSize = {50, 50}
                          --{1,0,  1,0,  1,0,  80,80  },
                          --{1,0,  1,0,  1,0,  20,20  },
                          --{1,0,  1,0,  1,0,  34,38  },
                        --[[{1,0,  2,3,  1,0  }, 
                        {1,0,  3,6,  1,0  }, 
                        {1,0,  7,6,  1,0  }, 
                        {1,0,  7,6,  7,6  }, 
                        {1,0,  10,9, 10,9 }, 
                        {5,8,  1,0,  1,0  }, 
                        {5,8,  5,4,  5,4  }, 
                        {5,8,  10,9, 10,9 } 
                       --]]
                                 
    --lock.waitUntilNoLocks()

    local nSkipped
    
    local nStartFrom = 1
    local all_NoisyLetterOpts_new = {}
    --print(#all_NoisyLetterOpts)
    for i = nStartFrom,#all_NoisyLetterOpts do
        table.insert(all_NoisyLetterOpts_new, all_NoisyLetterOpts[i])
    end
    --print(#all_NoisyLetterOpts_new)
    
    
    
    if onLaptop then

        do
            --error('!') 
            return
        end
        
    end
    
    repeat 
        
        nSkipped = doNoisyTrainingBatch(allNetworks, all_NoisyLetterOpts_new, allSNRs_test, nTrials)
            --function(fontNames, allSNRS_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)
    
        if (nSkipped > 0) and repeatUntilNoSkips then
            io.write(string.format('--------------------\n\n Skipped %d networks. Trying again in 2 minutes.\n\n----------------------\n', nSkipped))
            for i = 1, 120 do
                sys.sleep(1)
            end
        end
    until nSkipped == 0 or not repeatUntilNoSkips
  --]]  
  
end


        
       
    
    
--[[

if THREAD_ID and (math.mod(fi-1, N_THREADS_TOT)+1 ~= THREAD_ID) then
                print(string.format(' ** Thread %d, skipping font # %d ** \n', THREAD_ID, fi))
                
            else
                if THREAD_ID then
                    print(string.format(' ** Thread %d, doing font # %d ** \n', THREAD_ID, fi))
                end
            
            
            --]]
            
  --[[          
  (1): nn.SpatialConvolutionMap
  (2): nn.Tanh
  (3): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.Square
    (2): nn.SpatialSubSampling
    (3): nn.Sqrt
  }
  (4): nn.Reshape
  
  (5): nn.Linear
  (6): nn.Tanh
  
  (7): nn.Linear
  (8): nn.LogSoftMax

--]]
                                      --tbl_OriXY = allOriXY_mult, 
                                      --tbl_OriXY = allOriXY_scan,
                                      --tbl_OriXY = allOriXY_test,
                                      --tbl_OriXY = allOriXY_large_48,
                                      --tbl_OriXY = allOriXY_long,
                                      --tbl_OriXY = allOriXY_med1,
                                      
                                      
                                      --autoImageSize = true,
                                      --autoImageSize = false, imageSize = {32, 32},  
                                      --autoImageSize = false, imageSize = {40, 40},  
                                      --autoImageSize = false, imageSize = {32, 80},  
                                      --autoImageSize = false, imageSize = {45, 45},  
--                                      autoImageSize = false, imageSize = {65, 65},  
                                      --autoImageSize = false, imageSize = {40, 80},
                                      --autoImageSize = false, imageSize = {48, 48},  


                                      --autoImageSize = false, imageSize = {64, 64},                                        
                                      --autoImageSize = false, imageSize = {36, 88},  
                                      --autoImageSize = false, imageSize = {36, 164},  
                                      --autoImageSize = false, imageSize = {36, 116},
                                      --autoImageSize = false, tbl_imageSize = { {20, 20}, {50, 50}, },  --  {80, 80}
                                      --autoImageSize = false, tbl_sizeStyle_and_imageSize = { { 'k9', {35, 35}}, {'k18', {49, 49}}, {'k36', {99,99}}, {'k72', {147, 147} } },  --  {80, 80}
                                      --autoImageSize = false, tbl_sizeStyle_and_imageSize = { {'k18', {49, 49} },  {'k36', {99,99} }, {'k72', {147, 147} } },  --  {80, 80}
--                                      autoImageSize = false, tbl_sizeStyle_and_imageSize = { {'k18', {49, 49} },  {'k36', {99,99} }, {'k72', {147, 147} } },  --  {80, 80}
                                      --autoImageSize = false, tbl_sizeStyle_and_imageSize = {   {'k36', {99,99} },  },  --  {80, 80}
                                      --autoImageSize = false, tbl_sizeStyle_and_imageSize = { { 'k9', {35, 35}} },  --  {80, 80}




    --tbl_fontNames = { {'Bookman'}, Bookman_rBIJ,  mlhKsy_rBIJ}  }
    --tbl_fontNames = {  {'BookmanU'}, Bookman_rBIJ   }
    --tbl_fontNames = { Bookman_rBIJ, mK_rBIJ, mMlrhKsy_rBIJ }
    --tbl_fontNames = { {fonts = {'Bookman'}, styles = {'Roman'}},  {fonts = allStdFontNames, styles = {'Roman'}} }
    
    --tbl_fontNames = {  } 
    --tbl_fontNames = { allStdFontNames,  {fonts = allStdFontNames, styles = styles_use} }
    
    --tbl_fontNames = table.merge( { {allStdFontNames}, allStdFontNames })
    
   
    --------
    --- SIZE
    
        --tbl_sizeStyle = {'sml', 'med', 'big'}
    --tbl_sizeStyle = {'med'}
    --tbl_sizeStyle = {8, 16, 24,  12, 20,}
    --tbl_sizeStyle = {{24,8}, {16,8}, 8, 16, 24}
    --tbl_sizeStyle = {15, 16}
    --tbl_sizeStyle = {'k18'}
    --tbl_sizeStyle = {'k38'}
    --tbl_sizeStyle = {'k16'}
    --tbl_sizeStyle = {'k14'    }
    --tbl_sizeStyle = {'k30'}

    --allSNRs = {-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4}
    --local allSNRs = {0, 1, 2, 2.5, 3, 4}
    --local allSNRs = {0, 1, 1.5, 2, 2.5, 3, 4, 5}    
    --local allSNRs_band = {-3, -2, -1, 0, 1, 2, 3, 4, 5};
    
   
    --local nSNRs = #allSNRs
    --local allSNRs = {0, 1, 2,   2.5,   3, 4}
    
    
   
    
  
  
      --    local tbl_fontNames = allFontNames
--    local tbl_fontNames = allFontNames_ext
    
    --fontNames = {'Kuenstler', 'Yung'}
    --fontNames = {'Bookman', 'GeorgiaUpper'}
    --tbl_fontNames = {'Bookman'}
    --tbl_fontNames = { {'Bookman', 'BookmanB'} }
    --tbl_fontNames = { {'KuenstlerU', 'KuenstlerUB'} }
    --tbl_fontNames = {'Bookman', {'Bookman', 'BookmanB'} }
    --tbl_fontNames = {'Bookman', 'Braille', 'KuenstlerU'}

--    local allStdFontNames      = {'BookmanU', 'Sloan', 'HelveticaU', 'CourierU'};
    --local allStdFontNames      = {'BookmanU', 'HelveticaU'};
    --local allStdFontNames      = {'Bookman', 'BookmanU', 'Sloan', 'Helvetica', 'Courier', 'KuenstlerU', 'Braille', 'Yung'}
    --local allStdFontNames_tbl = { {'Bookman'}, {'BookmanU'}, {'Sloan'}, {'Helvetica'}, {'Courier'}, {'KuenstlerU'}, {'Braille'}, {'Yung'} }
    --local allStdFontNames      = {'Bookman', 'KuenstlerU', 'Braille'}
