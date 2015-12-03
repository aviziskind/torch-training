
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
            netStr_nice = string.format('(2 layers: %s HU)', toList(nHiddenUnits, ','))
        end
        return netStr, netStr_nice     
    end
    
    error('Incorrect input type')        
        
end





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
        networkOpts.poolTypes = table.rep(0, nConvLayers)
        
    else
        
        --- (1) poolSizes
        makeSureFieldIsCorrectLength('poolSizes')
        
        --- (2) poolStrides
        if networkOpts.poolStrides == 'auto' then
            networkOpts.poolStrides = networkOpts.poolSizes
        end
        
        makeSureFieldIsCorrectLength('poolStrides')
        
        --- (3) poolTypes        
        makeSureFieldIsCorrectLength('poolTypes')        
        
        -- if any layer has no pooling (poolSize == 0), set the stride & type to 0
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
    defaultParams.poolTypes = 2
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
    
    local nStates_str = abbrevList(networkOpts.nStatesConv, '_') 
    local nStates_str_nice = 'nStates=' .. table.concat(networkOpts.nStatesConv, ',') .. ';' 
    if nFCLayers > 0 then
        nStates_str  = nStates_str  ..  '_F' .. abbrevList(networkOpts.nStatesFC, '_')
        nStates_str_nice = nStates_str_nice .. 'FC=' .. table.concat(networkOpts.nStatesFC, ',') .. '; ' 
    end
    
    -- (1) filtsizes
    
    local filtSizes_str = ''
    local filtSizes_str_nice = '';
    assert(#networkOpts.filtSizes == nConvLayers)
    if not isequal_upTo(networkOpts.filtSizes, defaultParams.filtSizes, nConvLayers) then        
        if isequal_upTo(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str = '_nofilt'
        elseif length(table.nUnique(networkOpts.filtSizes)) == 1 then
            filtSizes_str = '_fs' .. networkOpts.filtSizes[1]
        else
            filtSizes_str = '_fs' .. abbrevList(networkOpts.filtSizes)
        end
    end
    
    networkPropertyStr = function(fieldName, networkOpts, defaultParams, fldAbbrev, str_NA)
        nConvLayers = (#networkOpts.nStatesConv)
        local networkProp = networkOpts[fieldName]
        local defaultProp = defaultParams[fieldName]
        local property_str = ''
        assert(#networkProp == nConvLayers)
        if not isequal_upTo(networkProp, defaultProp, nConvLayers) then        
            if isequal_upTo(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
                property_str = '_' .. str_NA
            elseif length(table.nUnique(networkOpts.filtSizes)) == 1 then
                property_str = '_' .. fldAbbrev .. networkProp[1]
            else
                property_str = '_' .. fldAbbrev .. abbrevList(networkProp)
            end
        end
        
        return property_str
    end
    
    
    
    filtSizes_str, filtSizes_str_nice = networkPropertyStr('filtSizes', networkOpts, defaultParams, 'fs', 'nofilt')
    
    
    if niceOutputFields == 'all' or table.contains(niceOutputFields, 'filtSizes') then
        if isequal(networkOpts.filtSizes, table.rep(0, nConvLayers), nConvLayers) then
            filtSizes_str_nice = ' No Filter.'
        else
            filtSizes_str_nice = ' FiltSz=' .. abbrevList(networkOpts.filtSizes, ',') .. '.'
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
    local poolTypes_str = ''
    local poolTypes_str_nice = ''
    local poolStrides_str = ''
    local poolStrides_str_nice = ''
    
    
    if skipAllPooling then
        doPooling_str = '_nopool'
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'doPooling') then
            doPooling_str_nice = ' No Pooling'
        end
        
    else
        -- assuming that the default is to do pooling.
        
        -- 2a. Pooling Present in each layer 
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'doPooling') then            
            doPooling_str_nice = ' Pooling: '
        end
        if nLayersWithPooling < nConvLayers then
                            
            for layer_i = 1, nConvLayers do
                local doPooling_i = (networkOpts.poolSizes[layer_i] == 0) 
                --doPooling_str = doPooling_str .. iff(doPooling_i, '_pool', '_nopool') 
                
                if niceOutputFields == 'all' or table.contains(niceOutputFields, 'doPooling') then    
                    --doPooling_str_nice = doPooling_str_nice .. iff(doPooling_i, 'Yes', 'No') .. iff(layer_i < nConvLayers, '/', '')
                end
            end
                
        end
                    
                
        -- 2b. Pool Size(s)
        if not isequal(networkOpts.poolSizes, defaultParams.poolSizes, nConvLayers) then
            poolSizes_str = '_psz' .. toTruncList(networkOpts.poolSizes, nConvLayers)
        end
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'poolSizes') then
            poolSizes_str_nice = ' PoolSz=' .. toTruncList(networkOpts.poolSizes, nConvLayers, ',') .. '.'
        end
        
        -- 2b. Pool Type(s) (pnorm)
        if not isequal(networkOpts.poolTypes, defaultParams.poolTypes, nConvLayers) then
            --print('filtSizes', networkOpts.filtSizes, filtSizes_default)
            if table.nUnique(networkOpts.poolTypes) > 1 then
                poolTypes_str = '_pt' .. toTruncList(networkOpts.poolTypes,nConvLayers)
            else
                poolTypes_str = '_pt' .. tostring(networkOpts.poolTypes[1])
            end
        end       
        if niceOutputFields == 'all' or table.contains(niceOutputFields, 'poolTypes') then            
            if table.nUnique(networkOpts.poolTypes) > 1 then
                poolTypes_str_nice = ' Pnorm=' .. toTruncList(networkOpts.poolTypes, nConvLayers, ',') .. '.'
            else
                poolTypes_str_nice = ' Pnorm=' .. tostring(networkOpts.poolTypes[1]) .. '.'
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
            poolStrides_str = '_pst' .. toTruncList(networkOpts.poolStrides, nConvLayers)
        end
        poolStrides_str_nice = ' PoolStrd=' .. toTruncList(networkOpts.poolStrides, nConvLayers, ',') .. '.'
        
    end
    
    
    local gpu_str = ''
    local useCUDAmodules = string.find(networkOpts.convFunction, 'CUDA')
    if networkOpts.trainOnGPU  then
        gpu_str = '_GPU'
        if useCUDAmodules and (GPU_BATCH_SIZE > 1) then
            gpu_str = '_GPU' .. tostring(GPU_BATCH_SIZE)
        end
    end
    
    

    local convNet_str      = convFcn_str .. nStates_str      .. filtSizes_str      .. doPooling_str      .. poolSizes_str      .. poolTypes_str      .. poolStrides_str .. gpu_str
    local convNet_str_nice = convFcn_str .. nStates_str_nice .. filtSizes_str_nice .. doPooling_str_nice .. poolSizes_str_nice .. poolTypes_str_nice .. poolStrides_str_nice  .. gpu_str
    return convNet_str, convNet_str_nice
    
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

    
