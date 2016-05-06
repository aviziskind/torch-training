getNetworkStr = function(networkOpts)
    
    local netStr, netStr_nice
    if networkOpts.netType == 'ConvNet' then
        local convStr, convStr_nice = getConvNetStr(networkOpts)
        netStr      = 'Conv_' .. convStr
        netStr_nice = 'ConvNet: ' ..  convStr_nice 
    elseif networkOpts.netType == 'MLP' then
        local MLP_str, MLP_str_nice = getMLPstr(networkOpts)    
        netStr      = 'MLP_' .. MLP_str
        netStr_nice = 'MLP: ' .. MLP_str_nice
        
    else
        error('Unknown network type : ' .. networkOpts.netType)
    end
    
    if networkOpts.trainConfig then
        local trainConfig_str = getTrainConfig_str(networkOpts.trainConfig)
        netStr = netStr .. trainConfig_str
    end
        
    if networkOpts.partModelOpts then  
        local partModelOpts_str = getPartModelOptStr(networkOpts.partModelOpts)  -- e.g. indep_90 (independent scales, 90x90 input)
        if partModelOpts_str ~= '' then
            netStr = netStr .. '_' .. partModelOpts_str
        end
    end
    
    if networkOpts.trialId and networkOpts.trialId ~= 1 then
        netStr = netStr .. '__tr' .. networkOpts.trialId
    end
        
        
    return  netStr, netStr_nice

end


getMLPstr = function(networkOpts) 

    local nHiddenUnits = networkOpts.nHiddenUnits
    if type(nHiddenUnits) == 'number' then
        nHiddenUnits = {nHiddenUnits}
    end
    
    local mlpStr = ''
    
    local nUnits_str, nUnits_str_nice
    if type(nHiddenUnits) == 'table' then
        if #nHiddenUnits == 0 then
            nUnits_str = 'X'            
        else
            nUnits_str = toList(nHiddenUnits)
        end
        
    end
    
    local gpu_str = getTrainOnGPUStr(networkOpts)
    
    local nLinType_str =  getNonlinearityStr(networkOpts)
    
    mlpStr      = nUnits_str  .. nLinType_str .. gpu_str
    
    return mlpStr, mlpStr
    --error('Incorrect input type')        
        
end


getNonlinearityStr = function(networkOpts)
    
    local nLinType_str, nLinType_str_nice = '', ''
    if networkOpts.nLinType then
       if string.lower(networkOpts.nLinType) == 'relu' then
           nLinType_str = '_rl'
           nLinType_str_nice  = 'ReLU'
       elseif string.lower(networkOpts.nLinType) == 'tanh' then
           --nLinType_str = 'th'
           nLinType_str_nice  = 'Tanh'
       else
            error(string.format('Unknown nonlinearity type : %s', networkOpts.nLinType));
       end
    end

    return nLinType_str, nLinType_str_nice
end

fixConvNetParams = function(networkOpts)
     
    local validSpatialNormTypes = {'none', 'gauss'}
    
    if (#networkOpts > 1) or networkOpts[1] then
        for j = 1,#networkOpts do
            networkOpts[j] = fixConvNetParams(networkOpts[j], defaultNetworkName)
        end
        return networkOpts
    end
            
     
   local defaultParams = getDefaultConvNetParams(networkOpts.defaultNet)
    --print(defaultNetworkName, defaultParams)
    
    local allowPoolStrideGreaterThanPoolSize = false
    --NetworkOpts = networkOpts
    
    local nStates, nStatesConv, nStatesFC
    local nConvLayers, nFCLayers
    if not networkOpts.nStatesConv then
        -- e.g input nStates = {6, 16, -120} ==> nStatesConv = {6, 16}, nStatesFC = {120}

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
        
    
    local makeSureFieldIsCorrectLength = function(fieldName, isValid_func)
        -- make sure is in table format (if was a number)
        
        if type(networkOpts[fieldName]) == 'number' or type(networkOpts[fieldName]) == 'string' then   -- num2tbl
            networkOpts[fieldName] = {networkOpts[fieldName]}
        end
                
        -- handle case where length is shorter than number of convolutional layers (repeat)
        local nInField = #(networkOpts[fieldName])
    
        if nInField < nConvLayers  then
            if nInField > 1 then
                error(string.format('%s: nConv = %d, nInField = %d', fieldName, nConvLayers, nInField));
            end
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

        
        if isValid_func then          
            for i = 1,nConvLayers do
                if not isValid_func( networkOpts[fieldName][i] ) then                    
                    error(string.format('Error for field %s : %s\n  (did not satisfy criteria function)', fieldName, networkOpts[fieldName][i] ) );
                end
            end
            
        end   
        
        
    end
    
    
    local isNumeric = function(x) return ( type(x) == 'number') end
    makeSureFieldIsCorrectLength('filtSizes', isNumeric)
    
    
    -- if any filtSizes == 0, set corresponding nStates equal to the number of states in the previous layer.
    
    for i = 1,nConvLayers do
        if networkOpts.filtSizes[i] == 0 then
            --print(io.write('setting state %d = %d', i, nStates_ext[i-1]))
            nStatesConv[i] = nStatesConv_ext[i-1]
        end        
    end
    

     -- (2) pooling  
    
    -- (2a) check if master flag is set to 0
    local skipAllPooling = (networkOpts.doPooling == false)     -- master value that overrides all other pooling values if set equal to false
    
    -- (2b) make sure all pooling parameters are the right length.    
    if skipAllPooling then
        networkOpts.poolSizes   = table.rep(0, nConvLayers)
        networkOpts.poolStrides = table.rep(0, nConvLayers)
        networkOpts.poolTypes   = table.rep(0, nConvLayers)
        
    else
        
        --- (1) poolSizes
        makeSureFieldIsCorrectLength('poolSizes', isNumeric)
        
        --- (2) poolStrides
        if networkOpts.poolStrides == 'auto' then
            networkOpts.poolStrides = networkOpts.poolSizes
        end
        
        --local isValidPoolStride = function(x) return ( type(x) == 'number' or x == 'AUTO' ) end        
        makeSureFieldIsCorrectLength('poolStrides', isNumeric)
        
        --- (3) poolTypes        
        local isValidPoolType = function(x) return ( type(x) == 'number' or x == 'MAX' ) end
        makeSureFieldIsCorrectLength('poolTypes', isValidPoolType)
        
        -- if any layer has no pooling (poolSize == 0 or 1), set the stride & type to 0
        for i = 1, nConvLayers do  
            if (networkOpts.poolSizes[i] == 0) or (networkOpts.poolSizes[i] == 1) then
                networkOpts.poolStrides[i] = 0
                networkOpts.poolTypes[i] = 0
            end
            if not allowPoolStrideGreaterThanPoolSize and (networkOpts.poolStrides[i] > networkOpts.poolSizes[i]) then
                networkOpts.poolStrides[i] = networkOpts.poolSizes[i]
            end
            
        end
        
    end    
    
    -- (2c) if no pooling at all, set the master flag to 0, and set all other parameters accordingly
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

 
       
    -- SUBTRACTIVE Normalization and    DIVISIVE Normalization

    local allSpatialNormTypes = {'Subtr', 'Div'}
    local validNormTypes = {'Gauss'};
    local isValidSpatialNormType = function(x) return table.anyEqualTo(x, validNormTypes) end
    for normt_i, normType in ipairs(allSpatialNormTypes) do
        
        local masterNormFlagField = 'doSpat' .. normType .. 'Norm'
        local normTypeField   = 'spat' .. normType .. 'NormType'
        local normWidthField  = 'spat' .. normType .. 'NormWidth'
        
    
        -- (3a) check if master flag is set to 0
        local skipAllNorm = (networkOpts[masterNormFlagField] == false) 
            or (networkOpts[normTypeField] == nil) or (networkOpts[normWidthField] == nil)
        
        
        --print(normType, skipAllNorm)
        -- (3b) check if master flag is set to 0
        if skipAllNorm then
            networkOpts[normTypeField]   = table.rep('none', nConvLayers)
            networkOpts[normWidthField]  = table.rep(0,      nConvLayers)
            
        else
            
            makeSureFieldIsCorrectLength(normTypeField,  isValidSpatialNormType)
            makeSureFieldIsCorrectLength(normWidthField, isNumeric)
        end
        
        
        -- (3c) if no normalization at all, set the master flag to 0, and set all other parameters accordingly
        local normInAnyLayers = false
        for i = 1,nConvLayers do
            if (networkOpts[normWidthField][i] > 0) then
                normInAnyLayers = true
            end
        end
        networkOpts[masterNormFlagField] = normInAnyLayers  and not skipAllNorm
        
        if not normInAnyLayers then
            networkOpts[normTypeField]   = table.rep('none', nConvLayers)
            networkOpts[normWidthField]  = table.rep(0,      nConvLayers)
        end
        
        
    end    
    
    
       
    return networkOpts
    
    
end





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
        
        
        
        
        
stringAndNumber = function(strWithNum)
--[[    
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
    --]]
    
    local idx_firstNum, idx_lastNum = string.find(strWithNum, '%d+')
    if not idx_firstNum then
        return strWithNum, nil
    else
        return string.sub(strWithNum, 1, idx_firstNum-1), 
               tonumber( string.sub(strWithNum, idx_firstNum, idx_lastNum) )
    end
            
            
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

    
getConvNetStr = function(networkOpts, niceOutputFields)
    --N0 = table.copy(networkOpts)
    local defaultParams = getDefaultConvNetParams(networkOpts.defaultNet)
    
    local defaultPoolStrideIsAuto = true
    
    niceOutputFields = niceOutputFields or 'all'
    
    --[[
    defaultParams.fanin = {1,4,16}
    defaultParams.filtSizes = {5,4}

    defaultParams.doPooling = true
    defaultParams.poolSizes = {4,2}
    defaultParams.poolTypes = 2
    defaultParams.poolStrides = {2,2}
    --]]    
    --print('----Before-------\n')
    --print(networkOpts)
    networkOpts = fixConvNetParams(networkOpts)
    defaultParams = fixConvNetParams(defaultParams)
    
    --print('----After-------\n\n\n')
    --print(networkOpts)
    
    local nConvLayers = (#networkOpts.nStatesConv)
    local nFCLayers = (#networkOpts.nStatesFC)

    local convFunction = networkOpts.convFunction
    --local trainOnGPU = networkOpts.trainOnGPU
    
    local convFcn_str = ''
    if convFunction == 'SpatialConvolution' then 
        convFcn_str = 'f'  -- f = 'fully connected'
    elseif convFunction == 'SpatialConvolutionMM' then 
        convFcn_str = 'm'  -- m = 'MM'    
    elseif convFunction == 'SpatialConvolutionMap' then 
        convFcn_str = ''
    elseif convFunction == 'SpatialConvolutionCUDNN' then 
        convFcn_str = 'cd'  
    elseif convFunction == 'SpatialConvolutionCUDA' then 
        convFcn_str = 'c'  -- c = 'CUDA
    else
        error(string.format('Unknown spatial convolution function : %s', tostring(convFunction)) )
    end
    
    local convPad_str = '';
    local convPad_str_nice = ''
    if networkOpts.zeroPadForConvolutions then -- default is : DON'T add zeroPadding before convolutions
        convPad_str = 'P'
        convPad_str_nice = '(P)'
    end
    
    local poolPad_str = '';
    local poolPad_str_nice = ''
    if networkOpts.zeroPadForPooling == false then   -- default is : DO add zeroPadding before pooling
        poolPad_str = 'NP'
        poolPad_str_nice = '(NP)'
    end
    
    
    
    
    local nStates_str = table.concat(networkOpts.nStatesConv, '_') 
    local nStates_str_nice = 'nStates=' .. table.concat(networkOpts.nStatesConv, ',') .. ';' 
    if nFCLayers > 0 then
        nStates_str  = nStates_str  ..  '_F' .. table.concat(networkOpts.nStatesFC, '_')
        nStates_str_nice = nStates_str_nice .. 'FC=' .. table.concat(networkOpts.nStatesFC, ',') .. '; ' 
    end
    
    local propertyOpts = {}
    propertyOpts.filtSizes = {prefix = 'fs', prefix_nice='FilterSz',  suffix=convPad_str,  suffix_nice = convPad_str_nice,
                              str_NA = 'nofilt', str_NA_nice = 'No Filter', is2D = true }
    propertyOpts.poolSizes = {prefix = 'psz', prefix_nice='PoolSize',  suffix=poolPad_str,  suffix_nice = poolPad_str_nice,
                              str_NA = 'nopool', str_NA_nice = 'No Pooling', is2D = true }
    propertyOpts.poolTypes = {prefix = 'pt', prefix_nice='Pnorm',  suffix='',  suffix_nice = '',  str_NA = '', str_NA_nice = '', is2D = false }
    propertyOpts.poolStrides = {prefix = 'pst', prefix_nice='PoolStrides',  suffix='',  suffix_nice = '',  str_NA = '', str_NA_nice = '', is2D = true }
    
    -- (1) filtsizes
    
    
    --[[
    local filtSizes_str = ''
    local filtSizes_str_nice = '';
    assert(#networkOpts.filtSizes == nConvLayers)
    if not isequal_upTo(networkOpts.filtSizes, defaultParams.filtSizes, nConvLayers) then        
        if isequal_upTo(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str = '_nofilt'
        elseif (table.nUnique(networkOpts.filtSizes)) == 1 then   --- XXX
            filtSizes_str = '_fs' .. networkOpts.filtSizes[1]
        else
            filtSizes_str = '_fs' .. abbrevList(networkOpts.filtSizes)
        end
    end
    
    --]]
    networkPropertyStr = function(fieldName)  -- fieldPrefix, fldAbbrev_nice, str_NA, str_NA_nice, is2D
        nConvLayers = (#networkOpts.nStatesConv)
        local networkProp = networkOpts[fieldName]
        local defaultProp = defaultParams[fieldName]
        local property_str = ''
        local property_str_nice = ''
        local opt = propertyOpts[fieldName]
        
        local extendProp = function(s)
            if opt.is2D then
                return s .. 'x' .. s
            else
                return tostring(s)
            end
        end
        
        assert(#networkProp == nConvLayers)
        if not isequal_upTo(networkProp, defaultProp, nConvLayers) then        
            if isequal_upTo(networkProp, table.rep(0, nConvLayers), nConvLayers) then
                property_str = '_' .. opt.str_NA
                property_str_nice =  opt.str_NA_nice .. '. '
                
            elseif (table.nUnique(networkProp)) == 1 then   --- XXX
                property_str = '_' .. opt.prefix .. networkProp[1] .. opt.suffix
                local prop_str_nice = extendProp(networkProp[1])
                property_str_nice = opt.prefix_nice .. '=' .. prop_str_nice  .. opt.suffix_nice .. '. '
                
            else
                local list_str, list_str_long
                if torch.type( networkProp[1] ) == 'string' then
                    list_str = table.concat( networkProp, '_')
                    list_str_long = list_str
                else
                    list_str = abbrevList(networkProp)
                    list_str_long = abbrevList(networkProp, '_', -1)
                end
                
                property_str = '_' .. opt.prefix .. list_str .. opt.suffix
                
                local tbl_vals = string.split(list_str_long, '_')
                local tbl_ext_vals = table.apply(extendProp, tbl_vals )
                local list_nice_ext = table.concat ( tbl_ext_vals, ',')
                
                property_str_nice =  opt.prefix_nice .. '=' .. list_nice_ext .. opt.suffix_nice .. '. '
                
            end
        end
        
        return property_str,property_str_nice
    end
    
    
        

    
    local filtSizes_str, filtSizes_str_nice = networkPropertyStr('filtSizes')
          
      
        --[[
    if niceOutputFields == 'all' or table.contains(niceOutputFields, 'filtSizes') then
        if isequal(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str_nice = ' No Filter.'
        else
            filtSizes_str_nice = ' FiltSz=' .. abbrevList(networkOpts.filtSizes, ',') .. '.'
        end
    end
    --]]

    -- (2) pooling
    local skipAllPooling =  networkOpts.doPooling == false
    local nLayersWithPooling = 0
    for i = 1,nConvLayers do
        if (networkOpts.poolSizes[i] > 0) and (not skipAllPooling) then 
            nLayersWithPooling = nLayersWithPooling + 1 
        end
    end
    skipAllPooling = skipAllPooling or (nLayersWithPooling == 0)
                    
    
    local doPooling_str,      poolSizes_str,      poolTypes_str,      poolStrides_str      = '', '', '', ''
    local doPooling_str_nice, poolSizes_str_nice, poolTypes_str_nice, poolStrides_str_nice = '', '', '', ''
    
    
    if skipAllPooling then
        doPooling_str = '_nopool'
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'doPooling') then
            doPooling_str_nice = ' No Pooling. '
        end
        
    else
        -- assuming that the default is to do pooling.
        
        -- 2a. Pooling Present in each layer 
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'doPooling') then            
            doPooling_str_nice = ' Pooling: '
        end
                
        -- 2b. Pool Size(s)
        poolSizes_str, poolSizes_str_nice = networkPropertyStr('poolSizes')
        
        -- 2b. Pool Types(s)
        poolTypes_str, poolTypes_str_nice = networkPropertyStr('poolTypes')

        -- 2c. PoolStrides(s)        
        assert(defaultPoolStrideIsAuto)
        if not isequal_upTo(networkOpts.poolSizes, networkOpts.poolStrides, nConvLayers) then
            poolStrides_str, poolStrides_str_nice = networkPropertyStr('poolStrides')
        end        

        
    end
    
    local nLinType_str, nLinType_str_nice =  getNonlinearityStr(networkOpts)
    
    local gpu_str = getTrainOnGPUStr(networkOpts)
    
    local dropout_str = getDropoutStr(networkOpts)
    
    local initializeLayers_str = getInitializeLayersStr(networkOpts)
    
    local convNet_str      = convFcn_str .. nStates_str      ..
            filtSizes_str  .. 
            doPooling_str  .. poolSizes_str      .. poolTypes_str      .. poolStrides_str ..  
            nLinType_str .. dropout_str .. gpu_str  .. initializeLayers_str
    
    local convNet_str_nice = convFcn_str .. nStates_str_nice .. 
            filtSizes_str_nice ..  
            doPooling_str_nice .. poolSizes_str_nice .. poolTypes_str_nice .. poolStrides_str_nice  .. 
            nLinType_str_nice .. dropout_str .. gpu_str .. initializeLayers_str
            
    return convNet_str, convNet_str_nice
    
end






getInitializeLayersStr = function(networkOpts)
    local initializeLayers_str = ''
        
    if networkOpts.initializeFirstLayer then
        initializeLayers_str = '__init_' .. networkOpts.autoEncoderNetworkName
        
    end    
    return initializeLayers_str
    
end


getTrainOnGPUStr = function(networkOpts)
    local gpu_str = ''
    --local useCUDAmodules = networkOpts.convFunction and string.find(networkOpts.convFunction, 'CUDA')
    if networkOpts.trainOnGPU  then
        --gpu_str = '_GPU'  --> now, only add this if batchSize > 1.
        if (networkOpts.gpuBatchSize and networkOpts.gpuBatchSize > 1) then
            gpu_str = '_GPU' .. tostring(networkOpts.gpuBatchSize)
        end
    end
    return gpu_str
end



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
        
getDropoutStr = function(networkOpts)
    
    local dropout_str = ''
    local dropout_default = -0.5
    if networkOpts.dropoutPs and not isequal(networkOpts.dropoutPs, {})  and not isequal(networkOpts.dropoutPs, 0) then
        if networkOpts.spatialDropout then
            dropout_str = '_Dr'
        else
            dropout_str = '_SDr'
        end
        if not isequal(networkOpts.dropoutPs, dropout_default) then 
            dropout_str = dropout_str .. toList(networkOpts.dropoutPs)
        end
    end
    
    return dropout_str
end


       
uniqueNetworks = function(allNetworks)

    local tbl_networkNames = {}
    local tbl_networkNames_nice = {}
    for i,net in ipairs(allNetworks) do
        local net_str, net_str_nice = getNetworkStr(net);
        
        table.insert(tbl_networkNames, net_str)
        table.insert(tbl_networkNames_nice, net_str_nice)
    end

    local uniqueNames, idx_unique = table.unique(tbl_networkNames)
    
    local uniqueNetworks = table.subsref(allNetworks, idx_unique)
    local uniqueNetworkNiceNames = table.subsref(tbl_networkNames_nice, idx_unique)
    return uniqueNetworks, uniqueNetworkNiceNames, uniqueNames

end


         
torch.isClassifierCriterion = function (criterionName)
    if criterionName == 'nn.ClassNLLCriterion' then
        return true
    elseif table.any( strcmp(criterionName, {'nn.MSECriterion', 'nn.WeightedMSECriterion', 'nn.mycri', 'nn.mycri_kai'})) then
        return false
    else
        error('Unknown criterion name')
    end

end


--[[
    -- SUBTRACTIVE Normalization
    local doSpatSubtrNorm = (networkOpts.doSpatSubtrNorm ~= false) and 
        networkOpts.spatSubtrNormType and networkOpts.spatSubtrNormType ~= 'none' and
        networkOpts.spatSubtrNormWidth and networkOpts.spatSubtrNormWidth > 0
    
    if networkOpts.doSpatSubtrNorm then
        networkOpts.doSpatSubtrNorm = true
    else
        networkOpts.doSpatSubtrNorm = false
        networkOpts.spatSubtrNormType = 'none'
        networkOpts.spatSubtrNormWidth = 0        
    end
    makeSureFieldIsCorrectLength('spatSubtrNormType')
    makeSureFieldIsCorrectLength('spatSubtrNormWidth')
        
        
        
    -- DIVISIVE Normalization
    local doSpatDivNorm = (networkOpts.doSpatDivNorm ~= false) and 
        networkOpts.spatDivNormType and networkOpts.spatDivNormType ~= 'none' and
        networkOpts.spatDivNormWidth and networkOpts.spatDivNormWidth > 0
    
    if networkOpts.doSpatDivNorm then
        networkOpts.doSpatDivNorm = true
    else
        networkOpts.doSpatDivNorm = false
        networkOpts.spatDivNormType = 'none'
        networkOpts.spatDivNormWidth = 0        
    end
    makeSureFieldIsCorrectLength('spatDivNormType')
    makeSureFieldIsCorrectLength('spatDivNormWidth')
    
   
    for i = 1,nConvLayers do
        if not table.any(strcmpi(networkOpts.spatSubtrNormType[i], {'none', 'gauss'})) then
            error(string.format('Unknown divisive normalization type : %s', networkOpts.spatSubtrNormType[i]))
        end
        if not table.any(strcmpi(networkOpts.spatDivNormType[i], {'none', 'gauss'})) then
            error(string.format('Unknown divisive normalization type : %s', networkOpts.spatDivNormType[i]))
        end
    end    
       --]]
       