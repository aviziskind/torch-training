doNoisyLettersTextureStats = function() -- (allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 
    print('Starting Noisy Texture Statistics Experiment')

    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

    local doBandNoiseExp = true


--    local fontNames_use = allFontNames_ext
    
    --fontNames = {'Kuenstler', 'Yung'}
    
    --local fontNames_use = allFontNames
    local fontNames_use = {'Bookman'}
    
    --fontNames_use = {'Sloan', 'KuenstlerU'}
    local allSNRs_test = allSNRs
    
    local allSNRs_band = {-1, 0, 1, 2, 3, 4};
    if doBandNoiseExp then
        allSNRs_test = allSNRs_band
    end
    
    --local netType = 'ConvNet'
    local netType = 'MLP'
    NetType = netType
    
    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 10   
    trainOpts.MAX_EPOCHS = 150
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1
    loadOpts.normalizeInputs = true

    -- size styles to do
    --local tbl_allSizeStyles = {'sml', 'med', 'big'}
    --local tbl_allSizeStyles = {'large'}
    --local tbl_allSizeStyles = {15, 16}
    --local tbl_allSizeStyles = {'med'}
    local tbl_allSizeStyles = {'k38'}

    --setName = '' -- 'poolSizes'
    --setName = 'filtSizes_poolSizes'
    
    --nTrials = 10
    
    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
    nTrials = 1
    local allNetworks
    local snr_train
    local allNStates, allNetNames 
    if netType == 'ConvNet' then
        -- ConvNet options

        --        snr_train = {1,2,3,4}
        snr_train = {1,2,3}

        
        --nstates = {30,80,600}	  --- x 5 --- 
    elseif netType == 'MLP' then
        --snr_train = {5}
        
--        local allNHiddenUnits = { {6}, {12}, {24}, {48}, {96} }
        --local allNHiddenUnits = { {}, {30}, {100} }
--        local allNHiddenUnits = { {}, {30} }
        local allNHiddenUnits = { {} }
            
  --          {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        local allMLPoptions = { netType = 'MLP', 
                                tbl_nHiddenUnits = allNHiddenUnits }
                    
        allNetworks = expandOptionsToList(allMLPoptions)
        
--        allNHiddenUnits = {4,10,20,40,100,200}
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}
        
    end
    
    --print(allNetworks)
 
 
    --local allSNRs_train = { {5}, {4}, {3}, {4,5}, {3,4}, {3,4,5}, {2,3,4,5}, {2,3,4}, {1, 2, 3, 4}, {1, 2, 3} }
    local tbl_allSNRs_train = {  {1,2,3,4} }
    
    local allBlurs = {0, 1, 1.5, 2, 3}
    
    local allBandNoise_cycPerLet = {0.5, 0.8, 1.3, 2.0, 3.2, 5.1, 8.1, 13}
    --local allBandNoise_cycPerLet = {0.5}
    local tbl_allBandNoise_cycPerLet = {}
    for i,v in ipairs(allBandNoise_cycPerLet) do
        tbl_allBandNoise_cycPerLet[i] = {filterType = 'band', cycPerLet_centFreq = v}
    end
    
    
    local allPinkNoise_exp = {1, 1.5, 2}    
    local tbl_allPinkNoise_exp = {}
    local tbl_allPinkOrWhiteNoise_exp = {}
    local tbl_allPinkPlusWhiteNoise_exp = {}
    for i,v in ipairs(allPinkNoise_exp) do
        tbl_allPinkNoise_exp[i]          = {filterType = '1/f', f_exp = v}
        tbl_allPinkOrWhiteNoise_exp[i]   = {filterType = '1/fOwhite', f_exp = v}
        tbl_allPinkPlusWhiteNoise_exp[i] = {filterType = '1/fPwhite', f_exp = v}
    end
    local whiteNoiseFilter = {filterType = 'white'}

    local tbl_allPinkNoises = table.merge(tbl_allPinkNoise_exp, whiteNoiseFilter)
        
    local allStatsUse = {'V1', 'V2'}
    --print('waiting for 2 hours... '); sys.sleep(3600*2)
    local all_NoisyLetterOpts_tbl = { stimType = 'NoisyLettersTextureStats',  
                                      Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  
                                      
                                      --autoImageSize = false, imageSize = {32, 32},  
                                      autoImageSize = false, imageSize = {64, 64},  
                                      
                                      Nscl_txt = 3, Nori_txt = 4, Na_txt = 5, 
                                      --tbl_statsUse = allStatsUse,
                                      statsUse = 'V2',
                                      
                                      --tbl_noiseFilter = tbl_allPinkNoise_exp,   
                                      
                                      --targetPosition = 2, nLetters = 3, Nx = 3, dX = 25,
                                      
                                      --tbl_noiseFilter = tbl_allBandNoise_cycPerLet, trainingNoise = 'same',
                                      tbl_noiseFilter = tbl_allBandNoise_cycPerLet, tbl_trainingNoise = tbl_allPinkNoises,
                                      --tbl_trainingNoise = tbl_allPinkNoises 
                                      --tbl_noiseFilter = tbl_allPinkNoises,   trainingNoise = 'same',
                                      
                                      --tbl_blurStd = allBlurs, autoImageSize = false, imageSize = {50, 50},
                                      tbl_SNR_train = tbl_allSNRs_train,
                                      tbl_sizeStyle = tbl_allSizeStyles 
                                    }
    
    local all_NoisyLetterOpts = expandOptionsToList(all_NoisyLetterOpts_tbl)
    

                   
    local noisyLetterOpts_strs = {}
    for i,opt in ipairs(all_NoisyLetterOpts) do
        noisyLetterOpts_strs[i] = getLetterOptsStr(opt)
    end
    print('Using the following noisy-letter settings : ');
    print(noisyLetterOpts_strs)
    
    doNoisyTrainingBatch(fontNames_use, allSNRs_test, allNetworks, all_NoisyLetterOpts, nTrials)
        --function(fontNames, allSNRS_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)
    
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
                local all_OriXY =  {  --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 2, Na_txt = 3, imageSize = {48,48}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 3, Na_txt = 5, imageSize = {48,48}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 5, imageSize = {32,32},  blurStd = 2,  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 }
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 4, Nori_txt = 4, Na_txt = 7, imageSize = {64,64}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 4, Nori_txt = 4, Na_txt = 9, imageSize = {64,64}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 }
--                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 5, imageSize = {64,64}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 }
                          
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 3, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 5, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 7, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          --{stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 9, imageSize = {32,32}, blurStd = 0,  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 9, imageSize = {32,32}, blurStd = 1,  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 9, imageSize = {32,32}, blurStd = 2,  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 3, Nori_txt = 4, Na_txt = 9, imageSize = {32,32}, blurStd = 3,  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 0, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 1, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 3, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 5, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 7, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          {stimType = 'NoisyLettersTextureStats',   Nscl_txt = 2, Nori_txt = 3, Na_txt = 9, Na_sub_txt = 9, imageSize = {32,32}, Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0 },
                          
                       }      
                       --]]
                       
                       