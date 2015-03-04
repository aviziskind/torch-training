

getNoisyLetterOptsStr = function(letterOpts)
        
    local oxyStr = getOriXYStr(letterOpts.OriXY)
    
    local targetPosition_str = ''       
    if letterOpts.targetPosition and letterOpts.targetPosition ~= 'all' then
        targetPosition_str = string.format('_T%d', letterOpts.targetPosition);
    end
    
    local nLetters_str = ''
    if letterOpts.nLetters and letterOpts.nLetters > 1 then
        nLetters_str = string.format('_L%d', letterOpts.nLetters);
    end
        
    local imageSizeStr = ''
    if not letterOpts.autoImageSize then
        imageSizeStr = string.format('-[%dx%d]', letterOpts.imageSize[1], letterOpts.imageSize[2]);        
    end
    
    local trainingImageSizeStr = ''
    if letterOpts.trainingImageSize and not isequal(letterOpts.trainingImageSize, 'same') 
            and not isequal(letterOpts.trainingImageSize, letterOpts.imageSize) then
        trainingImageSizeStr = string.format('_tr%dx%d', letterOpts.trainingImageSize[1], letterOpts.trainingImageSize[2]);        
    end

    
    local useBlur = letterOpts.blurStd and letterOpts.blurStd > 0
    local blurStr = ''
    if useBlur then
        blurStr = string.format('_blur%.0f', letterOpts.blurStd*10)
    end
    
    
    local trainingFonts_str = ''
    if letterOpts.trainingFonts and not isequal(letterOpts.trainingFonts, 'same') 
                                and not isequal(letterOpts.trainingFonts, letterOpts.fontName) then
        trainingFonts_str = '_trf' .. abbrevFontStyleNames(letterOpts.trainingFonts)
    end
    
    
    local trainingWiggle_str = ''
    if letterOpts.trainingWiggle and not isequal(letterOpts.trainingWiggle, 'same') 
                                 and not isequal(letterOpts.trainingWiggle, letterOpts.fontName.wiggles) then
        trainingWiggle_str = '_trW' .. getSnakeWiggleStr(letterOpts.trainingWiggle)
    end
    
    local trainingOriXY_str = ''
    if letterOpts.trainingOriXY and not isequal(letterOpts.trainingOriXY, 'same') 
                                and not isequal(letterOpts.trainingOriXY, letterOpts.OriXY) then
        trainingOriXY_str = '_trU' .. getOriXYStr(letterOpts.trainingOriXY)
    end
    
    
    local noiseFilter_str = noiseFilterOptStr(letterOpts)    -- includes "trained with" if appropriate

    local trainNoise_str = ''
    if letterOpts.trainingNoise and not isequal(letterOpts.trainingNoise, 'same') 
                                and not (filterStr(letterOpts.noiseFilter, 1) == filterStr(letterOpts.trainingNoise, 1)) then
        trainNoise_str = '_tr' .. filterStr(letterOpts.trainingNoise, 1)
        
    end
    
    
    assert(not (letterOpts.doOverFeat and letterOpts.doTextureStatistics))
    local textureStats_str = ''
    if letterOpts.doTextureStatistics then
        textureStats_str = getTextureStatsStr(letterOpts)    
    end
    
    local overFeat_str = '' 
    if letterOpts.doOverFeat then
        overFeat_str = getOverFeatStr(letterOpts)        
    end
    
    
    local retrainFromLayer_str = ''
    if letterOpts.retrainFromLayer and letterOpts.retrainFromLayer ~= '' then
        retrainFromLayer_str = '_rt' .. networkLayerStrAbbrev(letterOpts.retrainFromLayer)
    end
    
    local nPositions = letterOpts.Nori * letterOpts.Nx * letterOpts.Ny
    local indiv_pos_str = ''
    if letterOpts.trainOnIndividualPositions and (nPositions > 1) then
        indiv_pos_str = '_trIP'
    
        if letterOpts.retrainOnCombinedPositions then
            indiv_pos_str = indiv_pos_str .. '_rtCP'
        end
    end
                
                    
    local classifierForEachFont_str = ''
    local nFontShapes = (getFontClassTable(letterOpts.fontName)).nFontShapes
    if letterOpts.classifierForEachFont and (nFontShapes > 1) then
        classifierForEachFont_str = '_clsFnt'
    end    
                
                
    return oxyStr .. nLetters_str .. targetPosition_str .. imageSizeStr .. blurStr ..
            trainingOriXY_str .. trainingImageSizeStr .. trainingFonts_str .. trainingWiggle_str .. 
            noiseFilter_str .. trainNoise_str ..
            textureStats_str .. overFeat_str .. 
            retrainFromLayer_str .. indiv_pos_str .. classifierForEachFont_str
            
end



stringAndNumber = function(strWithNum)
    
    local maxForNum = 3
    local L = #strWithNum
    
    local str, num
    local numLastN
    for n = maxForNum, 1, -1 do
        local stringSub = string.sub(strWithNum, L-n+1, L)
        numLastN = tonumber(stringSub)
        if numLastN then
            str = string.sub(strWithNum, 1, L-n)
            num = numLastN
            return str, num
        end
    end
    
    -- no numbers at end - just return full string 
    return strWithNum, nil
            
end

networkLayerStrAbbrev = function(layerStrFull)
    
    local layerAbbrev_tbl = {conv='Conv',
                             linear= 'Lin',
                             classifier='Cls'}
    local layerStr, num = stringAndNumber(layerStrFull)
                                 
    local abbrev_str = layerAbbrev_tbl[layerStr]
    if not abbrev_str then
        error(string.format('%s is not one of the correct layers', layerStr))
    end
    assert(abbrev_str)
    if not num and layerStr ~= 'classifier' then
        num = 1
    end
    if num then
        abbrev_str = abbrev_str .. tostring(num)
    end
    return abbrev_str 
                     
end


getOriXYStr = function(OriXY)
        --print(OriXY)
    local oxyStr
    local dOri, nOri = OriXY.dOri, OriXY.Nori
    local dX, nX     = OriXY.dX,   OriXY.Nx
    local dY, nY     = OriXY.dY,   OriXY.Ny
    
    if nOri == 1 and nX == 1 and nY == 1 then
        oxyStr = '1oxy'
    else
        
        local ori_lims_str = ''
        local x_lims_str = ''
        local y_lims_str = ''
        
        if nOri > 1 then
            ori_lims_str = string.format('%dori[%s]_', nOri, dOri);
        end
        
        if nX > 1 then
            x_lims_str = string.format('%dx[%s]_', nX, dX);
        end
        
        if nY > 1 then
            y_lims_str = string.format('%dy[%s]_', nY, dY);
        end
        
        
        oxyStr = string.format('%s%s%s',  ori_lims_str, x_lims_str, y_lims_str);
        oxyStr = string.sub(oxyStr, 1, #oxyStr-1) -- remove trailing '_'
        
        
        --local ori_lims_str = getLimitsStr(nOri, letterOpts.ori_range);
        --local x_lims_str = getLimitsStr(nX, letterOpts.x_range);
        --local y_lims_str = getLimitsStr(nY, letterOpts.y_range);        
        --oxyStr = string.format('%dori%s_%dx%s_%dy%s', nOri,ori_lims_str, nX,x_lims_str, nY,y_lims_str);
    end
    
    return oxyStr
end



noiseFilterOptStr = function(letterOpts)
     
    local useNoiseFilter = letterOpts.noiseFilter 
    local noiseFilter_str = ''
    if useNoiseFilter then
        
        local testFilt_str = filterStr(letterOpts.noiseFilter)
        -- define training noise string
        
        -- final test noise filter:
        noiseFilter_str =  '_' .. testFilt_str
        if #noiseFilter_str == 1 then
            noiseFilter_str = ''
        end
        
    end
    return noiseFilter_str
    
end
 
   


filterStr = function(filt, wForWhite)
    
    local filtStr
    
    
    local applyFourierMaskGainFactor_default = false
    local applyFourierMaskGainFactor = applyFourierMaskGainFactor_default
    if filt and filt.filterType == 'white' then
        applyFourierMaskGainFactor = false
    elseif filt and filt.applyFourierMaskGainFactor then
        applyFourierMaskGainFactor = filt.applyFourierMaskGainFactor
    end
    local normStr = iff(applyFourierMaskGainFactor, 'N', '')
    
    
    if filt == nil or filt == 'same' then
        filtStr = ''
        
    elseif filt.filterType == 'white' then
        if wForWhite then
            filtStr = 'w'
        else
            filtStr = ''
        end
        
    elseif filt.filterType == 'band' then
        filtStr = string.format('Nband%.0f', filt.cycPerLet_centFreq*10)
        
    elseif filt.filterType == 'hi' then
        filtStr = string.format('Nhi%.0f', filt.cycPerLet_cutOffFreq*10)
    
    elseif filt.filterType == 'lo' then
        filtStr = string.format('Nlo%.0f', filt.cycPerLet_cutOffFreq*10)

    elseif filt.filterType == '1/f' then
        filtStr = string.format('Npink%.0f', filt.f_exp*10)
    
    elseif filt.filterType == '1/fPwhite' then
        filtStr = string.format('Npink%.0fPw', filt.f_exp*10)
    
    elseif filt.filterType == '1/fOwhite' then
        filtStr = string.format('Npink%.0fOw', filt.f_exp*10)
    else
        error(string.format('Unknown filter type: %s ', filt.filterType))
    end
    
    return filtStr .. normStr
    

end






getCrowdedLetterOptsStr = function(crowdedLetterOpts)
            
    local xrange = crowdedLetterOpts.xrange;
    
    assert(#xrange == 3)
    local x_range_str = string.format('x%d-%d-%d', xrange[1], xrange[2], xrange[3])
    
    
	local blur_str = ''
    local useBlur = crowdedLetterOpts.blurStd and  crowdedLetterOpts.blurStd > 0
    if useBlur then
        blur_str = string.format('_blur%.0f', crowdedLetterOpts.blurStd*10)
    end
    
    local imageSizeStr = ''
    if crowdedLetterOpts.imageSize then
        local sz = crowdedLetterOpts.imageSize
        imageSizeStr = string.format('-[%dx%d]', sz[1], sz[2]);        
    end
    
    
    local snr_str = get_SNR_str(crowdedLetterOpts.logSNR, '_', 1)
        
    local targetPositionStr = function(targetPosition)
        if type(targetPosition) == 'string' then
            assert(targetPosition == 'all')
            return 'T' .. targetPosition
        else
            return 'T' .. toOrderedList(targetPosition)
        end
    end
    
    
    local noiseFilter_str = noiseFilterOptStr(crowdedLetterOpts)    
    
    local details_str = ''
    
    local nLetters = 1
    if crowdedLetterOpts.nLetters then
        nLetters = crowdedLetterOpts.nLetters
    elseif crowdedLetterOpts.nDistractors then
        nLetters = crowdedLetterOpts.nDistractors + 1
    end        

--    if crowdedLetterOpts.nLetters and (crowdedLetterOpts.nLetters > 0) then
        
    local dnr_str = ''
    local distractorSpacing_str = ''
    local curTargetPosition_str
    local trainTargetPosition_str = ''
    
    local nLetters_str = string.format('%dlet', nLetters); 

    if nLetters == 1  then -- Training data (train on 1 letter)

        --targetPosition = crowdedLetterOpts.trainTargetPosition
        local trainTargetPosition = crowdedLetterOpts.trainTargetPosition or crowdedLetterOpts.targetPosition
        curTargetPosition_str = '_' .. targetPositionStr ( trainTargetPosition )


    elseif nLetters > 1 then -- Test on multiple letters

        local testTargetPosition = crowdedLetterOpts.testTargetPosition or crowdedLetterOpts.targetPosition
        curTargetPosition_str = '_' .. targetPositionStr ( testTargetPosition )
        
        dnr_str = string.format('_DNR%02.0f', crowdedLetterOpts.logDNR*10); -- distractor-to-noise ratio
        
        distractorSpacing_str = string.format('_d%d', crowdedLetterOpts.distractorSpacing); --  ie: all positions differences in X pixels
       
        if crowdedLetterOpts.testTargetPosition and crowdedLetterOpts.trainTargetPosition and not isequal(crowdedLetterOpts.trainTargetPosition, crowdedLetterOpts.testTargetPosition) then
            trainTargetPosition_str = string.format('_tr%s', targetPositionStr( crowdedLetterOpts.trainTargetPosition ) )
        end
                    
    end
    
    details_str = string.format('__%s%s%s%s', nLetters_str, distractorSpacing_str, dnr_str, trainTargetPosition_str);
        
    local textureStats_str = ''
    if crowdedLetterOpts.doTextureStatistics then
        textureStats_str = getTextureStatsStr(crowdedLetterOpts)
        assert(not crowdedLetterOpts.doOverFeat)
    end
    
    local overFeat_str = '' 
    if crowdedLetterOpts.doOverFeat then
        overFeat_str = getOverFeatStr(crowdedLetterOpts)        
        assert(not crowdedLetterOpts.doTextureStatistics)
    end
        
    return x_range_str .. curTargetPosition_str .. imageSizeStr .. snr_str .. blur_str .. noiseFilter_str .. textureStats_str .. overFeat_str .. details_str
    
end



    
getOverFeatStr = function(opts)
        
    local networkId_default = 0
    local layerId_default = 19
    
    local image_str = ''
    local networkId_str = ''
    local layerId_str = ''
    local contrast_str = ''
    
    local isImage = opts.OF_image == true
    if isImage then
        -- raw image file
        image_str = 'im'
    else    
        -- extracted features file
        local networkId = opts.networkId or networkId_default
        if networkId ~= networkId_default then
            networkId_str = string.format('_Net%d', networkId)
        end
        
        local layerId = opts.layerId or layerId_default
        if layerId ~= layerId_default then
            layerId_str = string.format('_L%d', layerId)
        end
            
        if opts.autoRescaleContrast then
            contrast_str = '_auto'
        else
            local contrast = opts.OF_contrast 
            local offset = opts.OF_offset
            contrast_str = string.format('_c%d_o%d', contrast, offset)
        end
    end
    
    return string.format('_OF%s%s%s%s', image_str, networkId_str, layerId_str, contrast_str)
        
end

getTextureStatsStr = function(letterOpts)
    
    local Nscl = letterOpts.Nscl_txt
    local Nori = letterOpts.Nori_txt
    local Na = letterOpts.Na_txt    
    
    
    local textureStats_str = ''
    if string.sub(letterOpts.textureStatsUse, 1, 2) == 'V2' then
        local useExtraV2Stats_str = iff(letterOpts.textureStatsUse == 'V2r', '_r', '')
 
        textureStats_str = string.format('_N%d_K%d_M%d%s', Nscl, Nori, Na, useExtraV2Stats_str)
    elseif string.sub(letterOpts.textureStatsUse, 1, 2) == 'V1' then
        
        textureStats_str = string.format('_N%d_K%d_%s', Nscl, Nori, letterOpts.textureStatsUse)
            
    end

    
    return textureStats_str

--[[
    local useSubsetOfA = letterOpts.Na_sub_txt and not (letterOpts.Na_sub_txt == 'all')  -- #(letterOpts.Na_sub_txt) > 0 
    local subsetChar = ''
    if useSubsetOfA then
        Na = letterOpts.Na_sub_txt;
        subsetChar = 'S';
    end
    local stat_params_str = string.format('_%dscl-%dori-%da%s',Nscl, Nori, Na, subsetChar)
--]]


end






--[[
getCrowdedLetterOptsStr = function(crowdedLetterOpts)
            
    local Nx = crowdedLetterOpts.Nx;
    local spacing = crowdedLetterOpts.spacing;
        
    return string.format('%dx_%s', Nx, spacing)
    
end
--]]


--[[
getCrowdedLetterOptsStr = function(crowdedLetterOpts, addTestStyle)
            
    local Nx = crowdedLetterOpts.Nx    
    local x_lims_str = getLimitsStr(Nx, crowdedLetterOpts.x_range);
    
    local nY = crowdedLetterOpts.nY    
    local y_lims_str = getLimitsStr(nY, crowdedLetterOpts.y_range);
    
    
    local basename = string.format('%dx%s_%dy%s', Nx,x_lims_str, nY,y_lims_str);
    local testStyle = string.format('%s%d', crowdedLetterOpts.targetPosition, crowdedLetterOpts.nDistractors); 
    
    if addTestStyle then
        basename = basename .. '_' .. testStyle
    end
    
    return basename, testStyle 

end
--]]

getTrainConfig_str = function(trainConfig)
    
    local logValueStr = function(x)
        if x > 0 then
            return tostring(-math.log10(x))
        else
            return 'X'
        end
    end
    local fracPart = function(x)
        local s = tostring(x)
        return string.sub(s, string.find(s, '[.]')+1, #s)
    end

    if trainConfig.adaptiveMethod then
        local adaptiveMethod = string.lower(trainConfig.adaptiveMethod);
        local isAdadelta = adaptiveMethod == 'adadelta'
        local isRMSprop  = adaptiveMethod == 'rmsprop'
        local isVSGD     = adaptiveMethod == 'vsgd'

        if isAdadelta or isRMSprop then
            local rho_default = 0.95
            local epsilon_default = 1e-6
            
            local rho = trainConfig.rho or rho_default
            
            local rho_str = ''
            if rho ~= rho_default then
                rho_str = '_r' .. fracPart(rho)
            end
            
            local epsilon = trainConfig.epsilon or epsilon_default
            local epsilon_str = ''
            if epsilon ~= epsilon_default then
                epsilon_str = '_e' .. logValueStr(epsilon)
            end
            if isAdadelta then
                return '__AD' .. rho_str .. epsilon_str
            elseif isRMSprop then
                return '__RMS' .. rho_str .. epsilon_str
            end
            
        elseif isVSGD then
            return '__vSGD' 
                
        end
    else
        local learningRate_default = 1e-3
        local learningRateDecay_default = 1e-4
        local momentum_default = 0 -- 0.95
        
        local learningRate = trainConfig.learningRate or learningRate_default        
        local learningRate_str = ''
        if learningRate ~= learningRate_default then
            learningRate_str = logValueStr (trainConfig.learningRate)
        end
        
        local learningRateDecay = trainConfig.learningRateDecay or learningRateDecay_default
        local learningRateDecay_str = ''
        if learningRateDecay ~= learningRateDecay_default then
            learningRateDecay_str = '_ld' .. logValueStr(trainConfig.learningRateDecay)
        end
        
        
        local momentum = trainConfig.momentum or momentum_default
        local momentum_str = ''
        if momentum ~= momentum_default then
            momentum_str = '_m' .. fracPart(trainConfig.momentum)
            
            if trainConfig.nesterov then
                momentum_str = momentum_str .. 'n'
            end
        end
        
        
        return '__SGD' .. learningRate_str  .. learningRateDecay_str .. momentum_str
        
    end
    
end
        
        


checkFontsFitInImage = function(fontNames, fontSize, imageSize)
    
    for fi,fontName in ipairs(fontNames) do
        local fontHeight = getFontAttrib(fontName, fontSize, 'height')
        local fontWidth = getFontAttrib(fontName, fontSize, 'width')
    
        local thisFontFits = fontHeight <= imageSize[1] and fontWidth <= imageSize[2]
        if not thisFontFits then
            print(string.format( '======= font = %d x %d is too large for imageSize = %dx%d ! =====', 
                fontHeight, fontWidth, imageSize[1], imageSize[2]) )
            return false
        end
    end
    
    return true
end
                    
checkNetworkFitsInImage = function(networkOpts, imageSize)
    
    local _, nOutputsEachBank = nOutputsFromConvStages(networkOpts, imageSize)
    
    local networkFits = nOutputsEachBank > 0
    --local net_str = getNetworkStr(networkOpts)
    --printf('\n --- net = %s. nOutputs = %d --- \n', net_str, nOutputsEachBank);
    if not networkFits then
        local net_str = getNetworkStr(networkOpts)
        print(string.format( '======= network (%s) is too small for imageSize = %dx%d ! =====', 
            net_str, imageSize[1], imageSize[2]) )
    end
    return networkFits
end
                    

getMetamerLetterOptsStr = function(metamerLetterOpts)
    return string.format('%dx%d_it%d', metamerLetterOpts.size, metamerLetterOpts.size, metamerLetterOpts.niter)    
end


getMLPstr = function(nHiddenUnits) 

    if type(nHiddenUnits) == 'number' then
        nHiddenUnits = {nHiddenUnits}
    end

    local netStr
    local netStr_nice
    if type(nHiddenUnits) == 'table' then
        if #nHiddenUnits == 0 then
            netStr = 'X'
            netStr_nice = '(1 layer)'
        else
            netStr = toList(nHiddenUnits)
            netStr_nice = string.format('(2 layers: %s HU)', toList(nHiddenUnits, nil, ','))
        end
        return netStr, netStr_nice     
    end
    
    error('Incorrect input type')        
        
end

getFontSizeStr = function(fontSize)
    if type(fontSize) == 'string' then
        return fontSize
    elseif type(fontSize) == 'number' then
        return tostring(fontSize)
    elseif type(fontSize) == 'table' then
        assert(#fontSize <= 2)
        if (#fontSize == 2) and (fontSize[1] == fontSize[2]) then
            return tostring(fontSize[1])
        else 
            return string.format('%d(%d)', fontSize[1], fontSize[2])
        end
    end
end


getExpSubtitle = function(letterOpts, networkOpts, trialId)
    
    local letterOpts_str = getLetterOptsStr(letterOpts)
    
    local network_str = '__' .. getNetworkStr(networkOpts)
    
    
    local trialId_str = ''
    if trialId and (trialId > 1) then
        trialId_str = '__tr' .. trialId
    end
        
    
    --local str = fontName_str .. sizeStyle_str .. snr_train_str .. letterOpts_str  .. network_str .. classesSep_str .. gpu_str .. trialId_str
    local str = letterOpts_str  .. network_str .. trialId_str
    return str
end

toList = function(X, maxN, sep)
    local typeX = getType(X)
    sep  = sep or '_'
    if typeX == 'table' then
        if #X == 0 then
            return ''
        end
        maxN = math.min(maxN or #X, #X)
        return table.concat(X, sep, 1, maxN)
        
    elseif typeX == 'number' then
        return tostring(X)
        
    --elseif string.find(typeX, 'Tensor') then
        
    elseif string.find(typeX, 'Storage') then
        
        maxN = math.min(maxN or #X, #X)
        local s = tostring(X[1])
        for i = 2, maxN do
            s = s .. sep .. tostring(X[i])
        end

        return s
    end
    error('Unhandled case type(X) = ' ..  typeX)
    
end


toOrderedList = function(X, maxN, sep, maxRunBeforeAbbrev)
    
    sep  = sep or '_'
    maxRunBeforeAbbrev = maxRunBeforeAbbrev or 2
    
    local abbrevSepValues = {[1] = 't', [0.5] = 'h', [0.25] = 'q', [5] = 'f', [10] = 'd'}

    local useHforHalfValues = true
    
    local typeX = getType(X)
    local str = ''
    
    if typeX == 'table' then
        if #X == 0 then
            return ''
        end
        maxN = math.min(maxN or #X, #X)
        
        
        local curIdx = 1
        str = tostring(X[1])
        while curIdx < maxN do
            local runLength = 0
            local initDiff = X[curIdx+1] - X[curIdx]
            local curDiff = initDiff
            while (curIdx+runLength < maxN) and (curDiff == initDiff) do
                runLength = runLength + 1
                if curIdx+runLength < maxN then
                    curDiff = X[curIdx+runLength+1] - X[curIdx+runLength]
                end
            end
            --print('run = ', runLength)
            if runLength >= maxRunBeforeAbbrev then
                --print('a');
                --print( 't' .. X[curIdx+runLength] )
                local abbrevSep
                for diffVal,diffSymbol in pairs(abbrevSepValues) do
                    
                    if initDiff == diffVal then
                        abbrevSep = diffSymbol
                    end
                end
                if not abbrevSep then
                    --print(initDiff)
                    abbrevSep = string.format('t%st', tostring(initDiff))
                end
                
                str = str .. abbrevSep .. tostring(X[curIdx+runLength])
                curIdx = curIdx + runLength+1
            else
                --print('b');
                --print( table.concat(X, sep, curIdx, curIdx+runLength) )
                if (runLength > 0) then 
                    str = str .. sep .. table.concat(X, sep, curIdx+1, curIdx+runLength)
                end 
                curIdx = curIdx + runLength+1
            end       
            if curIdx <= maxN then
                str = str .. sep .. tostring(X[curIdx])
            end
        end        
        
    elseif typeX == 'number' then
        str = tostring(X)
    else
        error('Unhandled case type(X) = ' ..  typeX)
    end
    
    str = string.gsub(str, '-', 'n')        
    
    if useHforHalfValues then
        str = string.gsub(str, '%.5', 'H')
    end
    
    return str
end



getLetterOptsStr = function(letterOpts)
        
    assert(type(letterOpts.fontName) == 'table')
    if (letterOpts.fontName[1] == 'SVHN') then
        return 'SVHN' .. getSVHNOptsStr(letterOpts.fontName.svhn_opts)
    end
        
    local fontName_str = abbrevFontStyleNames(letterOpts.fontName)
    
    local sizeStyle_str = '-' .. getFontSizeStr(letterOpts.sizeStyle)
    
    local snr_train_str = '_SNR' .. toOrderedList(letterOpts.SNR_train)
        
    local opt_str
    

    if table.anyEqualTo(letterOpts.expName, {'ChannelTuning', 'Complexity', 'Grouping', 'TrainingWithNoise', 'TestConvNet'})  then          
    --if table.any({'ChannelTuning', 'Complexity', 'Grouping', 'TrainingWithNoise'}, function(s) return (s == letterOpts.expName))
        opt_str = getNoisyLetterOptsStr(letterOpts)
    elseif letterOpts.expName == 'Crowding' then
        opt_str = getCrowdedLetterOptsStr(letterOpts)
    --elseif letterOpts.expName == 'NoisyLettersTextureStats' then
    --    opt_str = getNoisyLettersTextureStatsOptsStr(letterOpts)
    elseif letterOpts.expName == 'Metamer' then
        opt_str = getMetamerLetterOptsStr(letterOpts)
        
    else
        error(string.format('Unknown type: %s', letterOpts.expName))
    end
    
    
    return  fontName_str .. sizeStyle_str .. snr_train_str .. '__' .. opt_str
    
end

getSVHNOptsStr = function(svhn_opts)
    
    local useExtra_str = ''
    if svhn_opts.useExtraSamples then
        useExtra_str = 'x'
    end
    
    local globalNorm_str = ''
    if not svhn_opts.globalNorm then
        globalNorm_str = 'u' -- ="unnnormalized" (no global normalization) (default: global normalized)
    end
    
    local localContrastNorm_str = ''
    if svhn_opts.localContrastNorm then
        localContrastNorm_str = 'c' -- ="contrast" normalization (default: not contrast normalized)
    end
    
    local imageSize_str = ''
    local imageSize = {32, 32}
    if svhn_opts.imageSize then
        imageSize = svhn_opts.imageSize
    end
    if type(imageSize) == 'number' then
        imageSize = {imageSize, imageSize}
    end
    if imageSize[1] ~= 32 or imageSize[2] ~= 32 then
        imageSize_str = string.format('_%dx%d', imageSize[1], imageSize[2])
    end
    
    
    return globalNorm_str .. localContrastNorm_str .. useExtra_str .. imageSize_str
    
end





getSnakeWiggleStr = function ( wiggleSettings_orig )
   
    if not wiggleSettings_orig then
        return ''
    end
    local wiggleSettings = table.copy(wiggleSettings_orig)
   
    local isZero = function(x) return (x == 0) end
   
    local haveNoWiggle = wiggleSettings.none


    local haveOriWiggle = wiggleSettings.orientation
    if haveOriWiggle and type(wiggleSettings.orientation) ~= 'table' then
        wiggleSettings.orientation = {wiggleSettings.orientation}
    end 
    if haveOriWiggle and table.any(wiggleSettings.orientation, isZero) then
        wiggleSettings.orientation = table.nonzeros(wiggleSettings.orientation)
        haveNoWiggle = true
    end
    haveOriWiggle = haveOriWiggle and #wiggleSettings.orientation > 0
    
    
    local haveOffsetWiggle = wiggleSettings.offset
    if haveOffsetWiggle and type(wiggleSettings.offset) ~= 'table' then
        wiggleSettings.offset = {wiggleSettings.offset}
    end 
    
    if haveOffsetWiggle and table.any(wiggleSettings.offset, isZero) then
        wiggleSettings.offset = table.nonzeros(wiggleSettings.offset)
        haveNoWiggle = true
    end
    haveOffsetWiggle = haveOffsetWiggle and #wiggleSettings.offset > 0

    
    local havePhaseWiggle = wiggleSettings.phase
    if havePhaseWiggle and type(wiggleSettings.phase) ~= 'table' then
        wiggleSettings.phase = {wiggleSettings.phase}
    end 
    if havePhaseWiggle and table.any(wiggleSettings.phase, isZero) then
        wiggleSettings.phase = table.nonzeros(wiggleSettings.phase)
        haveNoWiggle = true
    end
    havePhaseWiggle = havePhaseWiggle and #wiggleSettings.phase > 0
    
    --local sep = '_'
    local sep = ''
    
    local str = ''
    if haveNoWiggle then
        local noWiggleStr = 'N'
        
        str = string.append(str, noWiggleStr, sep)
    end
    
    
    if haveOriWiggle then
        local oriAnglesStr = string.format('Or%s', toOrderedList(wiggleSettings.orientation))
        
        str = string.append(str, oriAnglesStr, sep)
    end

    if haveOffsetWiggle then
        local offsetAnglesStr = string.format('Of%s', toOrderedList(wiggleSettings.offset))
        
        str = string.append(str, offsetAnglesStr, sep)
    end
    
    if havePhaseWiggle then
        local phaseAnglesStr = 'Ph'
        
        str = string.append(str, phaseAnglesStr, sep)
    end
        
    return str
end


getNetworkStr = function(networkOpts)
    
    local netStr, netStr_nice
    if networkOpts.netType == 'ConvNet' then
        local convStr, convStr_nice = getConvNetStr(networkOpts)
        netStr      = 'Conv_' .. convStr
        netStr_nice = 'ConvNet: ' ..  convStr_nice 
    elseif networkOpts.netType == 'MLP' then
        local MLP_str, MLP_str_nice = getMLPstr(networkOpts.nHiddenUnits)    
        netStr      = 'MLP_' .. MLP_str
        netStr_nice = 'MLP: ' .. MLP_str_nice
        
    else
        error('Unknown network type : ' .. networkOpts.netType)
    end
    
    local trainConfig_str = '';
    if networkOpts.trainConfig then
        trainConfig_str = getTrainConfig_str(networkOpts.trainConfig)
        netStr = netStr .. trainConfig_str
    end
        
    return  netStr, netStr_nice

end


basename = function(fn, nrep)
    if not nrep then
        nrep = 1
    end
    local str = ''
    for i = 1,nrep do
        if i == 1 then
            str = paths.basename(fn)
        else
            str =  paths.basename(fn) .. '/' .. str
        end
        
        fn = paths.dirname(fn)
    end
    return str
end

expandOptionsToList = function(allOptions, loopKeysOrder)
    --print(allOptions)
    local baseTable = {}
    local loopKeys = {}
    local loopKeys_full = {}
    local loopValues = {}
    local nValuesEachLoopKey = {}
    local nTablesTotal = 1
    
    -- find which variables are to be looped over, and gather in a separate table
    for key_full,val in pairs(allOptions) do
        if string.sub(key_full, 1, 4) == 'tbl_' then
            local key = string.sub(key_full, 5, #key_full)
            
            table.insert(loopKeys_full, key_full)
            table.insert(loopKeys, key)
            
            --loopValues[keyName] = v
            if #val == 0 then
                print('val', val)
                print('options', allOptions)
                
                error(string.format('Key %s has 0 array entries\n', key))
            end
            
            
            table.insert(nValuesEachLoopKey, #val)
            nTablesTotal = nTablesTotal * (#val)
        else
            baseTable[key_full] = val
        end        
    end
    
    
    if loopKeysOrder then
        if type(loopKeysOrder) == 'string' then
            loopKeysOrder = {loopKeysOrder}
        end
        local idx_loopKeys_setOrder = {}
        for i,key in ipairs(loopKeysOrder) do
            local idx = table.find(loopKeys, key)
            if not idx then   error(string.format('No such field %s in table', key))  end
            idx_loopKeys_setOrder[i] = idx
        end
         
         local loopKeys_other_idxs = table.setdiff(table.range(1, #loopKeys), idx_loopKeys_setOrder)         
         --print('other=', loopKeys_other_idxs)
         local idx_new_order = table.merge(idx_loopKeys_setOrder, loopKeys_other_idxs)
                 
         loopKeys_full = table.subsref(loopKeys_full, idx_new_order)
         loopKeys = table.subsref(loopKeys, idx_new_order)
         nValuesEachLoopKey = table.subsref(nValuesEachLoopKey, idx_new_order)
        
            --print(loopKeys)
            --return;
    end
    
    -- initialize loop variables
    local nLoopFields = #nValuesEachLoopKey
    local loopIndices = {}
    for i = 1, nLoopFields do
        loopIndices[i] = 1
    end
    
    -- loop over all the loop-variables, assign to table.

    local allTables = {}
    for j = 1, nTablesTotal do
        local tbl_i = table.copy(baseTable)
        
        table.insert(allTables, tbl_i)
        
        if #loopIndices == 0 then
            break
        end
        
        for i = 1, nLoopFields do
            
            if type(allOptions[loopKeys_full[i]]) ~= 'table' then
                error(string.format('Field %s is not a table', loopKeys_full[i]))
            end
            local vals_field_i = table.copy( allOptions[loopKeys_full[i]] )
            
            if not string.find(loopKeys[i], '_and_') then
            
                tbl_i[loopKeys[i]] = vals_field_i[loopIndices[i]]
            else
                local sub_fields = {}
                for sub_fld in string.gmatch(string.gsub(loopKeys[i], '_and_', ','), "%a+") do 
                    table.insert(sub_fields, sub_fld)
                end                
                
                for k = 1,#sub_fields do
                    tbl_i[sub_fields[k]] = vals_field_i[loopIndices[i]][k];
                end
            end
        end
        
        local curFldIdx = 1
        
        loopIndices[curFldIdx] = loopIndices[curFldIdx] + 1
        while loopIndices[curFldIdx] > nValuesEachLoopKey[curFldIdx] do
            loopIndices[curFldIdx] = 1
            curFldIdx = curFldIdx + 1
            
            if curFldIdx > nLoopFields then
                assert(j == nTablesTotal)
                break;
            end
            loopIndices[curFldIdx] = loopIndices[curFldIdx]+1
        end
        
    end
    
    return allTables
    
end


testExpandOptionsToList = function()
    options_test = { netType = 'ConvNet', 
               poolStrides = 2,
               tbl_doPooling = {false, true},
               tbl_allNStates = { 1, 2 },
               --tbl_allNStates_and_test = { { 1 , 'B'}, {2, 'C'}  },
               tbl_filtsizes = { {6, 16}, {12, 32} }
               }

    list_test = expandOptionsToList(options_test)
    
    print('Initial options table');
    print(options_test)
    
    print('\n\nFinal list of tables:');
    for i,v in ipairs(list_test) do
        print(string.format('-- Table #%d --', i))
        print(v)
    end
    
end

--testExpandOptionsToList()
--[[
length = function(x)
    if type(x) == 'number' then
        return 1
    elseif type(x) == 'table' then
        return #x
    end
end
--]]



fixConvNetParams = function(networkOpts)
     
    if (#networkOpts > 1) or networkOpts[1] then
        for j = 1,#networkOpts do
            networkOpts[j] = fixConvNetParams(networkOpts[j])
        end
        return networkOpts
    end
            
     
    local defaultParams = getDefaultConvNetParams()
    
    local allowPoolStrideGreaterThanPoolSize = false
    --NetworkOpts = networkOpts
    
    local nStates, nStatesConv, nStatesFC
    local nConvLayers, nFCLayers
    if not networkOpts.nStatesConv then

        nStates = networkOpts.nStates
        networkOpts.nStatesConv = {}
        networkOpts.nStatesFC = {}
        for i,v in ipairs(nStates) do
            if v > 0 then
                table.insert(networkOpts.nStatesConv, v)
            elseif v < 0 then
                table.insert(networkOpts.nStatesFC, -v)
            end
        end    
        
    end
    
    nStatesConv = networkOpts.nStatesConv
    nStatesFC =   networkOpts.nStatesFC 

    nConvLayers = #nStatesConv
    nFCLayers   = #nStatesFC

    --local nStates = networkOpts.nStates
    --print(networkOpts)
    --print(nStates)
    local nStatesConv_ext = table.copy( nStatesConv )
    
    nStatesConv_ext[0] = 1   --- this assumes grayscale input (if rgb input, this should be 3)        
    --print('nConvLayers', nConvLayers)
        
    -- if there are any parameters not defined, assume they are the default parameters
    for k,v in pairs(defaultParams) do
        if (networkOpts[k] == nil) then
            networkOpts[k] = v
        end
    end
        
    
    local makeSureFieldIsCorrectLength = function(fieldName)
        -- make sure is in table format (if was a number)
        --NetworkOpts = networkOpts
        --FieldName = fieldName
        --print(networkOpts[fieldName])
        
        if type(networkOpts[fieldName]) == 'number' or type(networkOpts[fieldName]) == 'string' then   -- num2tbl
            networkOpts[fieldName] = {networkOpts[fieldName]}
        end
                
        -- handle case where length is shorter than number of convolutional layers (repeat)
        local nInField = #(networkOpts[fieldName])
    
        if nInField < nConvLayers  then
            assert(nInField == 1)
            networkOpts[fieldName] = table.rep(networkOpts[fieldName], nConvLayers, 1)
        end
    
        -- handle case where length is longer than number of convolutional layers (truncate)
        if nInField > nConvLayers then
            for j = nConvLayers+1,nInField do
                networkOpts[fieldName][j] = nil
            end
        end
        
        -- for 'max' pooling, make sure is uppercase ('MAX')
        for i = 1,nConvLayers do
            if type(networkOpts[fieldName][i]) == 'string' then
                networkOpts[fieldName][i] = string.upper(networkOpts[fieldName][i])
            end
        end

        
    end
    
    
    makeSureFieldIsCorrectLength('filtSizes')
    
    
    -- if any filtSizes == 0, set corresponding nStates equal to the number of states in the previous layer.
    
    for i = 1,nConvLayers do
        if networkOpts.filtSizes[i] == 0 then
            --print(io.write('setting state %d = %d', i, nStates_ext[i-1]))
            nStatesConv[i] = nStatesConv_ext[i-1]
        end        
    end
    

    -- (2) pooling
    local skipAllPooling = not networkOpts.doPooling 
    
        
    if skipAllPooling then
        networkOpts.poolSizes = table.rep(0, nConvLayers)
        networkOpts.poolStrides = table.rep(0, nConvLayers)
        networkOpts.poolType = table.rep(0, nConvLayers)
        
    else
        --- (1) poolSizes
        makeSureFieldIsCorrectLength('poolSizes')
        
        --- (2) poolStrides
        if networkOpts.poolStrides == 'auto' then
            networkOpts.poolStrides = networkOpts.poolSizes
        end
        
        makeSureFieldIsCorrectLength('poolStrides')
        
        --- (3) poolTypes        
        makeSureFieldIsCorrectLength('poolType')        
        
        -- if any layer has no pooling (poolSize == 0), set the stride & type to 0
        for i = 1, nConvLayers do  
            if (networkOpts.poolSizes[i] == 0) or (networkOpts.poolSizes[i] == 1) then
                networkOpts.poolStrides[i] = 0
                networkOpts.poolType[i] = 0
            end
            if not allowPoolStrideGreaterThanPoolSize and (networkOpts.poolStrides[i] > networkOpts.poolSizes[i]) then
                networkOpts.poolStrides[i] = networkOpts.poolSizes[i]
            end
            
        end
        
    end    
    
    local poolingInAnyLayers = false
    for i = 1,nConvLayers do
        if (networkOpts.poolSizes[i] > 0) then
            poolingInAnyLayers = true
        end
    end
    if not poolingInAnyLayers then
        networkOpts.poolSizes = table.rep(0, nConvLayers)
        networkOpts.poolStrides = table.rep(0, nConvLayers)
        networkOpts.poolTypes = table.rep(0, nConvLayers)
    end
    networkOpts.doPooling = poolingInAnyLayers

       
    return networkOpts
    
    
end

getConvNetStr = function(networkOpts, niceOutputFields)
    
    local defaultParams = getDefaultConvNetParams()
    
    local defaultPoolStrideIsAuto = true
    
    niceOutputFields = niceOutputFields or 'all'
    
    --[[
    defaultParams.fanin = {1,4,16}
    defaultParams.filtSizes = {5,4}

    defaultParams.doPooling = true
    defaultParams.poolSizes = {4,2}
    defaultParams.poolType = 2
    defaultParams.poolStrides = {2,2}
    --]]    
    --print('----Before-------\n')
    --print(networkOpts)
    networkOpts = fixConvNetParams(networkOpts)
    --print('----After-------\n\n\n')
    --print(networkOpts)
    
    local nConvLayers = (#networkOpts.nStatesConv)
    local nFCLayers = (#networkOpts.nStatesFC)

    local convFunction = networkOpts.convFunction
    --local trainOnGPU = networkOpts.trainOnGPU
    
    local convFcn_str = ''
    if convFunction == 'SpatialConvolutionMap' then -- or convFunction == 'SpatialConvolutionMM' then
        convFcn_str = ''
    elseif convFunction == 'SpatialConvolution' then 
        convFcn_str = 'f'  -- F = 'fully connected'
    elseif convFunction == 'SpatialConvolutionCUDA' then 
        convFcn_str = 'c'  -- F = 'fully connected'
    else
        error(string.format('Unknown spatial convolution function : %s', tostring(convFunction)) )
    end
    
    local nStates_str = table.concat(networkOpts.nStatesConv, '_') 
    local nStates_str_nice = 'nStates=' .. table.concat(networkOpts.nStatesConv, ',') .. ';' 
    if nFCLayers > 0 then
        nStates_str  = nStates_str  ..  '_F' .. table.concat(networkOpts.nStatesFC, '_')
        nStates_str_nice = nStates_str_nice .. 'FC=' .. table.concat(networkOpts.nStatesFC, ',') .. '; ' 
    end
    
    -- (1) filtsizes
    
    local filtSizes_str = ''
    local filtSizes_str_nice = '';
    if not isequal(networkOpts.filtSizes, defaultParams.filtSizes, nConvLayers) then        
        if isequal(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str = '_nofilt'
        else
            filtSizes_str = '_fs' .. toList(networkOpts.filtSizes, nConvLayers)
        end
    end
    if niceOutputFields == 'all' or tableContains(niceOutputFields, 'filtSizes') then
        if isequal(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str_nice = ' No Filter.'
        else
            filtSizes_str_nice = ' FiltSz=' .. toList(networkOpts.filtSizes, nConvLayers, ',') .. '.'
        end
    end

    -- (2) pooling
    local skipAllPooling = not networkOpts.doPooling
    local nLayersWithPooling = 0
    for i = 1,nConvLayers do
        if (networkOpts.poolSizes[i] > 0) and (not skipAllPooling) then 
            nLayersWithPooling = nLayersWithPooling + 1 
        end
    end
    skipAllPooling = skipAllPooling or (nLayersWithPooling == 0)
                    
    
    local doPooling_str = ''    
    local doPooling_str_nice = ''
    local poolSizes_str = ''
    local poolSizes_str_nice = ''
    local poolType_str = ''
    local poolType_str_nice = ''
    local poolStrides_str = ''
    local poolStrides_str_nice = ''
    
    
    if skipAllPooling then
        doPooling_str = '_nopool'
        if niceOutputFields == 'all' or tableContains(niceOutputFields, 'doPooling') then
            doPooling_str_nice = ' No Pooling'
        end
        
    else
        -- assuming that the default is to do pooling.
        
        -- 2a. Pooling Present in each layer 
        if niceOutputFields == 'all' or tableContains(niceOutputFields, 'doPooling') then            
            doPooling_str_nice = ' Pooling: '
        end
        if nLayersWithPooling < nConvLayers then
                            
            for layer_i = 1, nConvLayers do
                local doPooling_i = (networkOpts.poolSizes[layer_i] == 0) 
                --doPooling_str = doPooling_str .. iff(doPooling_i, '_pool', '_nopool') 
                
                if niceOutputFields == 'all' or tableContains(niceOutputFields, 'doPooling') then    
                    --doPooling_str_nice = doPooling_str_nice .. iff(doPooling_i, 'Yes', 'No') .. iff(layer_i < nConvLayers, '/', '')
                end
            end
                
        end
                    
                
        -- 2b. Pool Size(s)
        if not isequal(networkOpts.poolSizes, defaultParams.poolSizes, nConvLayers) then
            poolSizes_str = '_psz' .. toList(networkOpts.poolSizes, nConvLayers)
        end
        if niceOutputFields == 'all' or tableContains(niceOutputFields, 'poolSizes') then
            poolSizes_str_nice = ' PoolSz=' .. toList(networkOpts.poolSizes, nConvLayers, ',') .. '.'
        end
        
        -- 2b. Pool Type(s) (pnorm)
        if not isequal(networkOpts.poolType, defaultParams.poolType, nConvLayers) then
            --print('filtSizes', networkOpts.filtSizes, filtSizes_default)
            if table.nUnique(networkOpts.poolType) > 1 then
                poolType_str = '_pt' .. toList(networkOpts.poolType,nConvLayers)
            else
                poolType_str = '_pt' .. tostring(networkOpts.poolType[1])
            end
        end       
        if niceOutputFields == 'all' or tableContains(niceOutputFields, 'poolType') then            
            if table.nUnique(networkOpts.poolType) > 1 then
                poolType_str_nice = ' Pnorm=' .. toList(networkOpts.poolType, nConvLayers, ',') .. '.'
            else
                poolType_str_nice = ' Pnorm=' .. tostring(networkOpts.poolType[1]) .. '.'
            end
        
        end
        
        -- 2c. PoolStrides(s)
        
        assert(defaultPoolStrideIsAuto)
        --[[        
        local defaultPoolStrides = defaultParams.poolStrides
        if defaultPoolStrides == 'auto' then
            defaultPoolStrides = networkOpts.poolSizes -- use poolSize of current network
        end
        
        local currentPoolStrides = networkOpts.poolStrides
        if currentPoolStrides == 'auto' then
            currentPoolStrides = networkOpts.poolSizes
        end
        --]]
        if not isequal(networkOpts.poolSizes, networkOpts.poolStrides, nConvLayers) then
            poolStrides_str = '_pst' .. toList(networkOpts.poolStrides, nConvLayers)
        end
        poolStrides_str_nice = ' PoolStrd=' .. toList(networkOpts.poolStrides, nConvLayers, ',') .. '.'
        
    end
    
    
    local gpu_str = ''
    local useCUDAmodules = string.find(networkOpts.convFunction, 'CUDA')
    if networkOpts.trainOnGPU  then
        gpu_str = '_GPU'
        if useCUDAmodules and (GPU_BATCH_SIZE > 1) then
            gpu_str = '_GPU' .. tostring(GPU_BATCH_SIZE)
        end
    end
    
    

    local convNet_str      = convFcn_str .. nStates_str      .. filtSizes_str      .. doPooling_str      .. poolSizes_str      .. poolType_str      .. poolStrides_str .. gpu_str
    local convNet_str_nice = convFcn_str .. nStates_str_nice .. filtSizes_str_nice .. doPooling_str_nice .. poolSizes_str_nice .. poolType_str_nice .. poolStrides_str_nice  .. gpu_str
    return convNet_str, convNet_str_nice
    
end

    

tableContains = function(tbl, value)
    for k,v in pairs(tbl) do
        if v == value then
            return true
        end
    end
    return false

end



tblRep = function(x, n)
   local tbl = {}
   for i = 1,n do
       tbl[i] = x
   end
   return tbl
end

tblTrim = function(tbl, n)
    for i,v in ipairs(tbl) do
        if i > n then
            tbl[i] = nil
        end
    end
    return tbl
end

num2tbl = function(x)
    if type(x) == 'number' then
        return {x}
    elseif type(x) == 'table' then
        return x
    end
    error('Put in a number or a table');
end
        
        
uniqueNetworks = function(allNetworks)

    local tbl_networkNames = {}
    local tbl_networkNames_nice = {}
    for i,net in ipairs(allNetworks) do
        local net_str, net_str_nice = getNetworkStr(net);
        
        table.insert(tbl_networkNames, net_str)
        table.insert(tbl_networkNames_nice, net_str_nice)
    end

    local uniqueNames, idx_unique = tbl_unique(tbl_networkNames)
    
    local uniqueNetworks = table.subsref(allNetworks, idx_unique)
    local uniqueNetworkNiceNames = table.subsref(tbl_networkNames_nice, idx_unique)
    return uniqueNetworks, uniqueNetworkNiceNames, uniqueNames

end


uniqueOpts = function(allLetterOpts)

    local tbl_opts = {}
    for i,opt in ipairs(allLetterOpts) do
        local opt_str = getLetterOptsStr(opt)
        
        tbl_opts[i] = opt_str
    end

    local uniqueOptStrs, idx_unique = tbl_unique(tbl_opts)
    
    local uniqueOpts = table.subsref(allLetterOpts, idx_unique)

    return uniqueOpts, uniqueOptStrs

end

getFontAttrib = function(fontName, fontSize, param)
    -- created in MATLAB with 'createFontSizesFile.m'
    
    local fontSizeFile = matlabLetters_dir .. 'fontSizes.mat'
    local S = mattorch.load(fontSizeFile)
    
    local allSizes = S[fontName .. '_sizes']
    local idx_fontSize
    
    
    if (type(fontSize) == 'string') and (string.sub(fontSize, 1, 1) == 'k')  then
        local k_height = tonumber( string.sub(fontSize, 2, #fontSize) )
        local allKHeights = S[fontName .. '_k_heights']
        local k_idx_use
        for i = 1,allKHeights:numel() do
            if allKHeights[i][1] >= k_height then
                idx_fontSize = i
                break;
            end
        end           
        -- idx_fontSize = find(allKHeights, k_height)[1]
    else
        if (type(fontSize) == 'string') then  -- eg 'med' or 'big'
            fontSize = S[fontName .. '_' .. fontSize][1][1]
        
        --elseif (type(fontSize) == 'number') then        
        end
    
        idx_fontSize = find(allSizes, fontSize)[1]
    
    end    
    
    
    return  S[fontName .. '_' .. param .. 's'][idx_fontSize][1]        
    
end


--[[            
getAllDistractorSpacings = function(xrange, fontWidth, nDistractors)
    
    local letterSpacingPixels = 1
    local dx = xrange[2]
    local Nx = (xrange[3]-xrange[1])/dx + 1
--    Xrange = xrange
--    Dx = dx
--    NX = Nx
  
   
    assert(Nx == math.floor(Nx) and Nx == math.ceil(Nx))
    local allDistractSpacings_poss   = torch.range(1, (Nx-1)/nDistractors )
    
    --print(allDistractSpacings_poss)
    --print(torch.ge(allDistractSpacings_poss*dx, fontWidth + letterSpacingPixels))
    
    local allDistractSpacings_idx_use = torch.find(torch.ge(allDistractSpacings_poss*dx, fontWidth + letterSpacingPixels))
    
    local allDistractSpacings_pix_tnsr = allDistractSpacings_poss:index(1, allDistractSpacings_idx_use:long())*dx;
    
    local allDistractorSpacings_pix_tbl = torch.toTable(allDistractSpacings_pix_tnsr);
    
    return allDistractorSpacings_pix_tbl, allDistractSpacings_pix_tnsr
end
--]]

getAllDistractorSpacings = function(xrange, fontWidth, nDistractors, targetPosition)
    
    local letterSpacingPixels = 1
    local dx = xrange[2]
    local Nx = (xrange[3]-xrange[1])/dx + 1
    assert(Nx == math.floor(Nx) and Nx == math.ceil(Nx))
--    Xrange = xrange
--    Dx = dx
--    NX = Nx
    local minXSpacing = math.ceil( (fontWidth + letterSpacingPixels)/dx )
   
    local minTargetPos, maxTargetPos
    
    if type(targetPosition) == 'string' then
        assert(targetPosition == 'all') 
        minTargetPos, maxTargetPos = 1, Nx
    
    elseif type(targetPosition) == 'number' then
        minTargetPos, maxTargetPos = targetPosition, targetPosition
        
    elseif type(targetPosition) == 'table' then
        minTargetPos, maxTargetPos = table.max(targetPosition), table.min(targetPosition)
    
    elseif torch.isTensor(targetPosition) then
        minTargetPos, maxTargetPos = targetPosition:min(), table.min(targetPosition)        
    end
    
    local maxDistOnLeft  = minTargetPos - 1
    local maxDistOnRight = Nx - maxTargetPos

    local maxXSpacing
    if nDistractors == 1 then
        maxXSpacing = math.max(maxDistOnLeft, maxDistOnRight)
    elseif nDistractors == 2 then
        maxXSpacing = math.min(maxDistOnLeft, maxDistOnRight)
    end
    
    local allDistractSpacings = torch.range(minXSpacing, maxXSpacing)
    local allDistractSpacings_pix = allDistractSpacings * dx;
    
    return torch.toTable(allDistractSpacings_pix), allDistractSpacings_pix
    
end




abbrevFontStyleNames = function(names_orig, fontOrStyle)
    --print(names_orig)
   
    -- Book[M]an, Brai[L]le, [C]heckers4x4, Cou[R]ier, [H]elvetica, [K]uenstler  [S]loan, [Y]ung, 
    
    assert(names_orig)
    
    local skipAbbrevIfOneFont = true
    
    local allFontName_abbrevs = { Armenian =    'A',
                                  Bookman =     'M',
                                  Braille =     'L',
                                  Checkers4x4 = 'C',
                                  Courier =     'R',
                                  Devanagari =  'D', 
                                  Helvetica =   'H',
                                  Hebraica =    'I',
                                  Kuenstler =   'K',
                                  Sloan =       'S',
                                  Yung =        'Y',
                                  
                                  SVHN =        'N',
                                  
                                  Snakes =      'G',   -- '=Gabor'
                                 }
    
    local allFontName_abbrevs_med = { Armenian    = 'Arm',
                                      Bookman     = 'Bkm',
                                      Braille     = 'Brl',
                                      Checkers4x4 = 'Ch4',
                                      Courier     = 'Cur',
                                      Devanagari  = 'Dev', 
                                      Helvetica   = 'Hlv',
                                      Hebraica =    'Heb',
                                      Kuenstler   = 'Kun',
                                      Sloan       = 'Sln',
                                      Yung        = 'Yng',
                                      
                                      SVHN = 'SVHN', 
                                      
                                      Snakes = 'Snk',
                                 }
                                 
                             
    local allStyleName_abbrevs = { Roman      = 'r',
                                   Bold       = 'B',
                                   Italic     = 'I',
                                   BoldItalic = 'J', 
                                 }
    
    local allStyleName_abbrevs_med = { Roman      = 'Rom',
                                       Bold       = 'Bld',
                                       Italic     = 'Ital',
                                       BoldItalic = 'BldItal', 
                                     }
     
    
    
    local tbl_short = {}
    local tbl_med = {}
    local tbl_full = {}
    
    if type(names_orig) == 'string' then
        names_orig = {names_orig}
    end
    
    
    if names_orig.fonts or names_orig.styles or names_orig.wiggles or names_orig.svhn_opts then
       local fontName_str_short,   fontName_str_medium,  fontNames_full   = '', '', ''
       local styleName_str_short,  styleName_str_medium, styleNames_full  = '', '', ''
       local pre_full, join_short, join_med, join_full,  post_full        = '', '', '', '', ''
       local opt_str = ''
       if names_orig.fonts then
           --print(names_orig.fonts)
            fontName_str_short, fontName_str_medium, fontNames_full = abbrevFontStyleNames(names_orig.fonts, 'font')
       end
       
       if (names_orig.styles) and ((#names_orig.styles== 0) or (#names_orig.styles == 1  and names_orig.styles[1] == 'Roman')) and names_orig.fonts then
           names_orig.styles = nil
       end
       
       if names_orig.styles then
            styleName_str_short, styleName_str_medium, styleNames_full = abbrevFontStyleNames(names_orig.styles, 'style')
       end
       
       if names_orig.wiggles then            
            local wiggle_str = getSnakeWiggleStr(names_orig.wiggles) 
            styleName_str_short, styleName_str_medium, styleNames_full = wiggle_str, wiggle_str, wiggle_str
       end
       
       if names_orig.svhn_opts then            
            local svhn_opt_str = getSVHNOptsStr(names_orig.svhn_opts) 
            styleName_str_short, styleName_str_medium, styleNames_full = svhn_opt_str, svhn_opt_str, svhn_opt_str
       end
       
       if names_orig.fonts and (names_orig.styles or names_orig.wiggles) then
           --pre_full, join_short, join_med, join_full, post_full = '{', '_', '_x_', ' X ', '}'
           join_short, join_med = '_', '_x_'
           pre_full, join_full, post_full = '{', ' X ', '}'
       end
       --print(names_orig)
       
       if names_orig.opts then
           opt_str = getFontOptStr(names_orig.opts)
       end
       
       return fontName_str_short .. join_short .. styleName_str_short .. opt_str, 
              fontName_str_medium .. join_med  .. styleName_str_medium .. opt_str,
              pre_full .. fontNames_full .. join_full .. styleNames_full .. opt_str .. post_full
       
       
    end
    
    local names = table.copy(names_orig)
   
    
    if not fontOrStyle then
        --print(names_orig[1])
        if allFontName_abbrevs[ getRawFontName( names[1]) ] then
            fontOrStyle = 'font'
        elseif allStyleName_abbrevs[ names[1] ] then
            fontOrStyle = 'style'
        else
            error('!')
        end
    end
    
   
   
    if skipAbbrevIfOneFont and (fontOrStyle == 'font') and (#names == 1) then
        return names[1], names[1], names[1]
    end

    
    
    if fontOrStyle == 'font' then
        table.sort(names)
        
        
    elseif fontOrStyle == 'style' then
        local styleOrder = {'Roman', 'Bold', 'Italic', 'BoldItalic'}
        local names = table.reorderAs(names, styleOrder);
        names = names_new
        
    elseif fontOrStyle == 'wiggle' then
        names = {getSnakeWiggleStr(names)}
        
    end
    

    local str_full = table.concat(names, ',');

    --getRawFontName(

    for i,name in ipairs(names) do
        
        local abbrev_short, abbrev_med
        if fontOrStyle == 'font' then
            local rawFontName, fontAttrib = getRawFontName(name)
            local bold_str = iff(fontAttrib.bold_tf, 'B', '')
            local italic_str = iff(fontAttrib.italic_tf, 'I', '')
            local upper_str = iff(fontAttrib.upper_tf, 'U', '')
        
            local fontAbbrev     = allFontName_abbrevs[rawFontName]
            local fontAbbrev_med = allFontName_abbrevs_med[rawFontName]
            if not fontAttrib.upper_tf then                
                fontAbbrev = string.lower(fontAbbrev)
            end
            abbrev_short = fontAbbrev ..                  bold_str .. italic_str
            print(abbrev_short)
            abbrev_med   = fontAbbrev_med .. upper_str .. bold_str .. italic_str
            print(abbrev_med)
        elseif fontOrStyle == 'style' then
            
            local styleName = name
            abbrev_short = allStyleName_abbrevs[styleName]
            abbrev_med   = allStyleName_abbrevs_med[styleName]
            
        elseif fontOrStyle == 'wiggle' then
            abbrev_short = names[1]
            abbrev_med = names[1]
            
        end
        
        --tbl_short[i] = fontAbbrev .. upper_str .. bold_str
        tbl_short[i] = abbrev_short
        tbl_med[i]   = abbrev_med 
        
    end
    local str_short = table.concat(tbl_short)
    local str_medium = table.concat(tbl_med, '_')
    
    return str_short, str_medium, str_full
    
    
end



getFontList = function(fontTable)
    
    if type(fontTable) == 'string' then   -- if just input a single font name  ("Bookman")
        fontTable = {fontTable}
    end
    if not fontTable.fonts and not fontTable.styles and not fontTable.wiggles then  -- already have a list of fonts
        return fontTable
    end
            
    if not fontTable.fonts then
        error('Input table has a "styles" field, but no "fonts" field')
    end
    
    
    local fontNameList = fontTable.fonts 
    if type(fontNameList) == 'string' then   -- if just input a single font name  ("Bookman")
        fontNameList = {fontNameList}
    end
    
    local fonts_styles_tbl, stylesAbbrev
    
    if fontTable.styles then
        stylesAbbrev = table.copy(fontTable.styles)
        local stylesAbbrevTable = {Roman = '', Bold = 'B', Italic = 'I', BoldItalic = 'BI'}
        for i,styleFull in ipairs(stylesAbbrev) do
            stylesAbbrev[i] = stylesAbbrevTable[styleFull]
        end
    elseif fontTable.wiggles then
            
        local wiggleList = getWiggleList(fontTable.wiggles)
        stylesAbbrev = {}
        for i,w in ipairs(wiggleList) do
            stylesAbbrev[i] = getSnakeWiggleStr(w)
        end
        
        --print(wiggleList)
        --print(stylesAbbrev)
        
    else
        
        stylesAbbrev = {''}
    end
    fonts_styles_tbl = expandOptionsToList({tbl_font = fontNameList, tbl_style = stylesAbbrev})
    
    local fonts_styles = {}       
    for i = 1, #fonts_styles_tbl do
        fonts_styles[i] = fonts_styles_tbl[i].font .. fonts_styles_tbl[i].style
    end                        
    return fonts_styles
    
end                    
                    

getWiggleList = function(wiggleSettings)
    local wiggleList = {}
    if wiggleSettings.none then
        table.insert(wiggleList, {none = 1})
    end
    
    if wiggleSettings.orientation then
        for i,ori in ipairs(totable(wiggleSettings.orientation)) do
            table.insert(wiggleList, {orientation = ori})
        end
    end
    
    if wiggleSettings.offset then
        for i,off in ipairs(totable(wiggleSettings.offset)) do
            table.insert(wiggleList, {offset = off})
        end
    end
    
    if wiggleSettings.phase then
        table.insert(wiggleList, {phase = 1})
    end

    return wiggleList
end

tbl_max = function(tbl)
    local max_val = tbl[1]
    for k,v in pairs(tbl) do
        max_val = math.max(max_val, v)
    end
    return max_val
end



getNumClassesForFont = function(fontName, sumIfMultipleFonts)
    
    if type(fontName) == 'table' then
       local nClasses = {}
       local nClassesTot = 0
       for i,font in ipairs(fontName) do
            nClasses[i] = getNumClassesForFont(font)
            nClassesTot = nClassesTot + nClasses[i]
       end
       if sumIfMultipleFonts then
           return nClassesTot
       else
           return nClasses
       end
    end
    
    
    local font = getRawFontName(fontName)
    if (font=='Bookman') or (font=='Courier') or (font=='Helvetica') or (font=='Kuenstler') 
        or (font== 'Sloan') or (font== 'Yung') or (font== 'Devanagari') or (font== 'Braille') or (font== 'Checkers4x4') then
        return 26
    elseif (font== 'Hebraica') then
        return 22
    elseif (font== 'Armenian') then
        return 35
        
        
    elseif (font== 'SVHN') or (font == 'Snakes') then
        return 10
    else
        error(string.format('Unknown font : %s', font))
    end
    
    
end




getRawFontName = function(fontName, keepUpperCaseFlag)

    local rawFontName = fontName
    local fontAttrib = {upper_tf = false, bold_tf = false, italic_tf = false}
    if not fontName or #rawFontName == 0 then
        return '', false, false, false
    end   
    
    local LastChar = function(s) return string.sub(s, #s, #s) end
    local lastChar = LastChar(rawFontName)
    
    while string.find( 'UBI', lastChar ) or tonumber(lastChar) do
        
        if lastChar == 'U' then
            fontAttrib.upper_tf = true
        elseif lastChar == 'B' then
            fontAttrib.bold_tf = true; 
        elseif lastChar == 'I' then
            fontAttrib.italic_tf = true;
        elseif lastChar == 'U' and keepUpperCaseFlag then
            break;
        end
        
        
        if tonumber(lastChar) then
            local i = #rawFontName;
            while tonumber(string.sub(rawFontName, i, i)) and i > 1  do
                i = i-1;
            end
            local n = tonumber(string.sub(rawFontName, i+1, #rawFontName));
            if n > 10 then
                n = n / 10;
            end
            local tp = string.sub(rawFontName, i, i)
            if tp == 'O' then
                fontAttrib.outline_w = n;
            elseif tp == 'T' then
                fontAttrib.thin_w = n;
            end
            rawFontName = string.sub(rawFontName, 1, i)
            
        end
        
        rawFontName = string.sub(rawFontName, 1, #rawFontName - 1)
        lastChar = LastChar(rawFontName)
    end
    
    if string.find(rawFontName, 'Snakes') then
        rawFontName = 'Snakes'
    end    
    
    return rawFontName, fontAttrib
end




getFontClassTable = function(fontNamesSet)
    local fontNamesList = table.copy(getFontList(fontNamesSet))
    table.sort(fontNamesList)
   --[[
    Kuenstler = 0
    Bookman = 26
    BookmanU = 52
    Braille = 78
    nClasses = 104
   --]]
    local nClassesTot = 0
    local nFontShapes = 0
    local classesTable = {}
   
    for fi, fontName in ipairs(fontNamesList) do
        local fontName_raw = getRawFontName(fontName, 1)
       
        if not classesTable[fontName_raw] then
            local nClasses = getNumClassesForFont(fontName_raw)
                
            classesTable[fontName_raw] = nClassesTot
            nClassesTot = nClassesTot + nClasses
            nFontShapes = nFontShapes + 1
        end
    end
    classesTable.nClassesTot = nClassesTot 
    classesTable.nFontShapes = nFontShapes
    
    return classesTable
end





fileExistsInPreferredSubdir = function(mainDir, preferredSubdir, filename)
    --local maxLevelDown = maxLevelDown or 1
    if not preferredSubdir then
        preferredSubdir = ''
    end
            
    local preferredFileName = mainDir .. preferredSubdir .. filename
    local alsoCheckNYU_folder = true
    local mainDir_NYU

    
    if paths.filep(preferredFileName) then
        return true, preferredFileName
    elseif alsoCheckNYU_folder and not string.find(mainDir, 'NYU') then
        mainDir_NYU = string.sub(mainDir, 1, #mainDir-1) .. '_NYU/'
        local preferredFileName_NYU = mainDir_NYU .. preferredSubdir .. filename
        if paths.filep(preferredFileName_NYU) then
            return true, preferredFileName_NYU
        end
    end
    
    local altFileName = mainDir .. filename
    if paths.filep(altFileName) then
        return true, altFileName
    end
        
    
    --local dir_name = paths.dirname(filename_full)
    --local file_name = paths.basename(filename_full)
    
    --[[
    local sub_dirs_str = sys.execute(string.format('ls -d %s*/', mainDir))
    local subdirs = string.split(sub_dirs_str)
    --]]
    local subdirs = subfolders(mainDir)
    if alsoCheckNYU_folder and not string.find(mainDir, 'NYU') and paths.dirp(mainDir_NYU) then
        subdirs = table.merge(subdirs, subfolders(mainDir_NYU))
    end
        
    for i,subdir in ipairs(subdirs) do
        local file_name_i = subdir .. filename
        
        if paths.filep(file_name_i) then
            return true, file_name_i
        end
    end
        
    return false, preferredFileName
    
end



getBestWeightsAndOffset = function(inputMatrix, labels, nClasses)
    local nSamples = inputMatrix:size(1)
    local h = inputMatrix:size(3)
    local w = inputMatrix:size(4)
    
    assert(labels:numel() == nSamples)
    
    nClasses = nClasses or labels:max()
    local nEachClass = torch.Tensor(nClasses):zero()
    local E1 = torch.Tensor(nClasses)
    local templates = torch.Tensor(nClasses, 1, h, w):zero()
    for i = 1,nSamples do
        local class_idx = labels[i]
        templates[labels[i]] = templates[labels[i]] + inputMatrix[i]
        nEachClass[labels[i]] = nEachClass[labels[i]] + 1        
    end
    
    for i = 1, nClasses do
        templates[i] = templates[i] / nEachClass[i]
        
        E1[i] = torch.dot(templates[i], templates[i])
    end
    
    local bias = -E1/2
    
    return templates, bias
    
    
end


torch.copyTensorValues = function(t1, t2)
    T1 = t1
    T2 = t2
    local iter1 = torch.tensorIterator(t1)
    local iter2 = torch.tensorIterator(t2)
    
    local s1 = t1:storage()
    local s2 = t2:storage()
    
    while true do
        local i1, v1 = iter1()
        local i2, v2 = iter2()
        
        if not v1 or not v2 then
            break
        end
        --printf('[%d(%.1f)->%d(%.1f)]\n', i1, v1, i2, v2)
        s2[i2] = v1
        
    end
    
    
end


torch.tensorIterator = function(x)
    
    local nDims = x:nDimension()
    local sizeX = x:size()
    local idxs = torch.Tensor(nDims):fill(1)
    local storage = x:storage()
    local stride = x:stride()
    local nElementsInTensor = x:numel()
    local nElementsInStorage = #storage
    
    local storageOffset = x:storageOffset()
    
    return function()
    
        -- get index of current element
        local curStorageIdx = storageOffset
        for dim = 1,nDims do
            curStorageIdx = curStorageIdx + (idxs[dim] - 1)*stride[dim]
        end
        
        if curStorageIdx > nElementsInStorage then
            return 
        end
                
        local curElement = storage[curStorageIdx]
        
        -- increment the idxs appropriately along each dimension
        local dim = nDims
        idxs[dim] = idxs[dim]+1
        while (idxs[dim] > sizeX[dim]) and (dim > 1) do
            idxs[dim] = 1
            idxs[dim-1] = idxs[dim-1]+1
            dim = dim-1            
        end
        
        return curStorageIdx, curElement
        --print('idxs = ', idxs1[1], idxs1[2])
        
    end
    
    
end

iterateOverTensor = function(X)
    for i,v in torch.tensorIterator(X) do 
        printf('%d : %.1f\n', i,v)
    end    
    
end


getDataSubfolder = function(letterOpts)
    local subfolder
    
    subfolder = letterOpts.stimType
    --[[
    if letterOpts.expName == 'Crowding' then
        subfolder = 'Crowding/' .. subfolder
    end
    --]]

    --[[
    if letterOpts.expName == 'ChannelTuning' or letterOpts.expName == 'Complexity' then
        subfolder = letterOpts.stimType        
    else
        error('Unhandled case')
    end 
    --]]
    
    return subfolder
end


checkFolderExists = function(dirname)
    if string.find(  string.sub(dirname, #dirname - 5, #dirname), '[.]') then  -- ie. has filename at end
        dirname = paths.dirname(dirname)
    end
    if not paths.dirp(dirname) then
        error(string.format('Error: Path does not exist: %s', dirname))
    end        
end





    




--[[

--getNoisyLettersTextureStatsOptsStr = function(letterOpts)
  
    
    local imageSize = letterOpts.imageSize
    local imageSize_str = string.format('%dx%d', imageSize[1],imageSize[2])
    
    local Nscl = letterOpts.Nscl_txt
    local Nori = letterOpts.Nori_txt
    local Na = letterOpts.Na_txt
      
    local useSubsetOfA = letterOpts.Na_sub_txt and not (letterOpts.Na_sub_txt == 'all')  -- #(letterOpts.Na_sub_txt) > 0 
    local subsetChar = '';
    if useSubsetOfA then
        Na = letterOpts.Na_sub_txt;
        subsetChar = 'S';
    end
    local stat_params_str = string.format('_%dscl-%dori-%da%s',Nscl, Nori, Na, subsetChar)


    local useBlur = letterOpts.blurStd and letterOpts.blurStd > 0
    local blur_str
    if useBlur then
        blur_str = string.format('_blur%.0f', letterOpts.blurStd*10)
    else
        blur_str = ''
    end
        
    local statsParams_str = ''
    if string.sub(letterOpts.textureStatsUse, 1, 2) == 'V2' then
        local useExtraV2Stats_str = iff(letterOpts.textureStatsUse == 'V2r', '_r', '')
 
        statsParams_str = string.format('_%dscl-%dori-%da%s%s', Nscl, Nori, Na, subsetChar, useExtraV2Stats_str)
    elseif string.sub(letterOpts.textureStatsUse, 1, 2) == 'V1' then
        
        statsParams_str = string.format('_%dscl-%dori_%s', Nscl, Nori, letterOpts.textureStatsUse)
            
    end
    
    local noiseFilter_str = noiseFilterOptStr(letterOpts)

    return imageSize_str .. statsParams_str  .. noiseFilter_str  .. blur_str


    --test_str = iff(isfield(letterOpts, 'tf_test') && isequal(letterOpts.tf_test, 1), '_test', '');
    
    --return  imageSizeStr .. stat_params_str .. blurStr
 

end
--]]

   --[[
    local wiggleType = wiggleSettings.wiggleType;
    local wiggleAngle = wiggleSettings.wiggleAngle;
    
    if wiggleType == 'none' then
        wiggleAngle = 0
    end    

    if wiggleAngle == 0 then
        return ''
    end
    
    if wiggleType == 'orientation' then 
        return string.format('Or%d', wiggleAngle)
    elseif wiggleType == 'offset' then
        return string.format('Of%d', wiggleAngle)
    elseif wiggleType == 'phase' then
        return string.format('Ph')
    end
--]]
