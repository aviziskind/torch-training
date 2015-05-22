require 'image'  -- for normalization kernels

generateModel = function(inputStats, networkOpts, letterOpts)
    
    local nInputs = inputStats.nInputs
    local height  = inputStats.height;        assert(height)
    local width   = inputStats.width;         assert(width)
    local nClasses = inputStats.nClasses or inputStats.nOutputs;  assert(nClasses)
    local nPositions = inputStats.nPositions or 1
    
    local useConvNet = networkOpts.ConvNet or networkOpts.netType == 'ConvNet'
    local doSplit = (networkOpts.split and networkOpts.split == true) or false
                
        
    --nInputSide = math.sqrt(networkOpts.nInputs)
        
    --doSplit = (splitAfterLayer ~= nil)
    local nUnitsInLastLayer
    local feat_extractor = nn.Sequential()
    local classifier     = nn.Sequential()
    local classifier_indiv_pos          = nn.Sequential()
    local classifier_indiv_pos_combined = nn.Sequential()
        
        
    local fontClassesTable = nil
    if letterOpts and letterOpts.classifierForEachFont then        
        fontClassesTable = getFontClassTable(getFontList(letterOpts.fontName))
        nClasses = fontClassesTable.nClassesTot
    end
        
    local params = {}
    --input layer 
    if networkOpts.netType == 'MLP'  then

        local nHiddenUnitsEachLayer = networkOpts.nHiddenUnits
        
        if type(nHiddenUnitsEachLayer) == 'number' then
            nHiddenUnitsEachLayer = {nHiddenUnitsEachLayer}
        end
        
        feat_extractor:add(  nn.Reshape(nInputs) )
        nUnitsInLastLayer = nInputs
        
        -- hidden layers (if any)
        
        for layer_i,nUnitsInThisLayer in ipairs(nHiddenUnitsEachLayer) do
            feat_extractor:add(  nn.Linear(nUnitsInLastLayer, nUnitsInThisLayer) )
            feat_extractor:add(  nn.Tanh() )
            
            nUnitsInLastLayer = nUnitsInThisLayer
        end
            
        --output layer
        classifier:add(  nn.Linear(nUnitsInLastLayer, nClasses) )
        classifier:add(  nn.LogSoftMax() )
        
        if nPositions > 1 then
            local indiv_pos_classifier = nn.Linear(nUnitsInLastLayer, nClasses * nPositions);
            
            classifier_indiv_pos:add( indiv_pos_classifier )  -- classifier for each individual position
            classifier_indiv_pos:add( nn.LogSoftMax() )
            
            classifier_indiv_pos_combined:add(  indiv_pos_classifier )  -- classifier for each letter, taking in inputs from each letter at each individual position
            classifier_indiv_pos_combined:add(  nn.LogSoftMax() )
            classifier_indiv_pos_combined:add(  nn.Exp() )
            classifier_indiv_pos_combined:add(  nn.Linear(nClasses * nPositions, nClasses) )
            classifier_indiv_pos_combined:add(  nn.LogSoftMax() )
        end
        
        --classifier:add(  nn.Reshape(nClasses) )
        params.nHiddenUnitsEachLayer = nHiddenUnitsEachLayer
        params.resizeInputToVector = true
        --reshapeImage = true
        
    elseif networkOpts.netType == 'ConvNet' then

        networkOpts = fixConvNetParams(networkOpts)
        --NetworkOpts = networkOpts
        
        local nInputPlanes = 1 
        
        local nStatesConv = networkOpts.nStatesConv
        nStatesConv[0] = nInputPlanes
        local nConvLayers = #nStatesConv
        --nStates_copy2 = nStatesConv
        
        --print(nStatesConv)
        --local useConnectionTable_default = true
        --local useConnectionTable_default = false
        --local useConnectionTable = useConnectionTable_default and not trainOnGPU
        --params.enforceStridePoolSizeEqual = trainOnGPU
        local n = networkOpts
        local convFunction, fanin, filtSizes, doPooling, poolSizes, poolTypes, poolStrides, trainOnGPU = 
            n.convFunction, n.fanin, n.filtSizes, n.doPooling, n.poolSizes, n.poolTypes, n.poolStrides, n.trainOnGPU
        
        local doSpatSubtrNorm,   spatSubtrNormType,   spatSubtrNormWidth,   doSpatDivNorm,   spatDivNormType,   spatDivNormWidth = 
            n.doSpatSubtrNorm, n.spatSubtrNormType, n.spatSubtrNormWidth, n.doSpatDivNorm, n.spatDivNormType, n.spatDivNormWidth
            
           
        
        if convFunction == 'SpatialConvolutionMap' and trainOnGPU then
            error('SpatialConvolutionMap cannot be trained on the GPU ...')
        end
        
        local useCUDAmodules = string.find(convFunction, 'CUDA')
                
        if trainOnGPU then
            assert(isequal(poolStrides, poolSizes))
        end
        
        
        local connectTables = {}
        local nOut_conv_h = {}
        local nOut_conv_w = {}
        local nOut_pool_h = {}
        local nOut_pool_w = {}
        nOut_pool_h[0] = height
        nOut_pool_w[0] = width
        
        
        if useCUDAmodules then
            feat_extractor:add(nn.Transpose({1,4},{1,3},{1,2}))  -- transpose so that batch dim is last (CUDA modules expect nPlanes x h x w x batchSize)
        end        
        
        for layer_i = 1,nConvLayers do
            -- 1. Convolutional layer
            local SpatConvModule
            local kW, kH = filtSizes[layer_i], filtSizes[layer_i]
            if (filtSizes[layer_i] > 0) then 
                
                if convFunction == 'SpatialConvolutionMap' then
                    -- fanin: how many incoming connections (from a state) in previous layer each output unit receives input from.
                    -- this cant be more than the number of states in previous layer!
                    Fanin = fanin
                    NStatesConv = nStatesConv
                    Layer_i = layer_i
                    local fanin_use = math.min(fanin[layer_i], nStatesConv[layer_i-1]) 
                    connectTables[layer_i] = nn.tables.random(nStatesConv[layer_i-1], nStatesConv[layer_i], fanin_use) -- OK b/c only 1 feature--- 
                    SpatConvModule = nn.SpatialConvolutionMap(connectTables[layer_i], kW, kH)
                    
                elseif convFunction == 'SpatialConvolution' then
                    SpatConvModule = nn.SpatialConvolution(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH)
                    
                elseif convFunction == 'SpatialConvolutionCUDA' then
                    SpatConvModule = nn.SpatialConvolutionCUDA(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH)
                    
                else 
                    error('Unknown spatial convolution function : ' .. tostring(convFunction))
                end
                feat_extractor:add(SpatConvModule)
                
                nOut_conv_h[layer_i] = nOut_pool_h[layer_i-1] - filtSizes[layer_i] + 1
                nOut_conv_w[layer_i] = nOut_pool_w[layer_i-1] - filtSizes[layer_i] + 1

                -- 2. Nonlinearity (sigmoid)
                feat_extractor:add(nn.Tanh())

                
            elseif (filtSizes[layer_i] == 0) then   -- use filtSize == 0, as flag to skip convolution step (& tanh) in this layer
                nStatesConv[layer_i] = nStatesConv[layer_i-1]
                
                nOut_conv_h[layer_i] = nOut_pool_h[layer_i-1]
                nOut_conv_w[layer_i] = nOut_pool_w[layer_i-1]

                --feat_extractor:add(nn.Tanh())

            end
                
            
            -- 3. Spatial pooling / sub-sampling
            local poolType_thisLayer = poolTypes[layer_i]
            if doPooling and not (poolType_thisLayer == 0) then
                
                local doMaxPooling = (type(poolType_thisLayer) == 'string') and (string.upper(poolType_thisLayer) == 'MAX')
                local poolingModule = nil
                local kW, kH = poolSizes[layer_i],   poolSizes[layer_i]
                local dW, dH = poolStrides[layer_i], poolStrides[layer_i]
                
                if useCUDAmodules then
                    if doMaxPooling then
                        poolingModule = nn.SpatialMaxPoolingCUDA( kW, kH, dW, dH )
                    else
                        error('non-max pooling not implemented for cuda yet')
                    end
                else
                
                    if doMaxPooling then
                        poolingModule = nn.SpatialMaxPooling( kW, kH, dW, dH )
                    elseif poolType_thisLayer == 1 then
                        poolingModule = nn.SpatialL1Pooling(nStatesConv[layer_i], kW, kH, dW, dH )
                    elseif poolType_thisLayer == 2 then                    
                        poolingModule = nn.SpatialLPPooling(nStatesConv[layer_i], poolType_thisLayer, kW, kH, dW, dH)
                    else                     
                        error(string.format('unhandled case: pool type = %s', tostring(poolType_thisLayer)))
                        --feat_extractor:add(nn.Power(poolTypes))
                        --feat_extractor:add(nn.SpatialSubSampling(nStatesConv[layer_i], 
                          --      poolSizes[layer_i], poolSizes[layer_i], poolStrides[layer_i], poolStrides[layer_i]))
                        --feat_extractor:add(nn.Power(1/poolTypes))
                    end
                end
                assert(poolStrides[layer_i] <= poolSizes[layer_i])
                
                feat_extractor:add(poolingModule)
                
                nOut_pool_h[layer_i] = math.floor( (nOut_conv_h[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
                nOut_pool_w[layer_i] = math.floor( (nOut_conv_w[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
            else
                nOut_pool_h[layer_i] = nOut_conv_h[layer_i]
                nOut_pool_w[layer_i] = nOut_conv_w[layer_i]            
            end
         
 
            local doSpatSubtrNorm_thisLayer = doSpatSubtrNorm and string.lower(spatSubtrNormType[i]) ~= 'none'
            if doSpatSubtrNorm_thisLayer then
                local norm_kernel = getNormKernel(spatSubtrNormType[i], spatSubtrNormWidth[i])
                feat_extractor:add( nn.SpatialSubtractiveNormalization(nStatesConv[layer_i], norm_kernel ) )
            end
            
            local doSpatDivNorm_thisLayer   = doSpatDivNorm  and string.lower(spatDivNormType[i]) ~= 'none' 
            if doSpatDivNorm_thisLayer then
                local norm_kernel = getNormKernel(spatDivNormType[i], spatDivNormWidth[i])
                feat_extractor:add( nn.SpatialSubtractiveNormalization(nStatesConv[layer_i], norm_kernel ) )
            end
            --]]
            
        end
        
        if useCUDAmodules then
            feat_extractor:add(nn.Transpose({4,1},{4,2},{4,3}))        
        end

        local nOutputs_last = nStatesConv[nConvLayers] * (nOut_pool_h[nConvLayers]*nOut_pool_w[nConvLayers])
        
        if nOutputs_last <= 0 then
            error('No outputs for linear layers')
        end
        --[[
        print('nOut_conv_h', nOut_conv_h)
        print('nOut_conv_w', nOut_conv_w)
        
        print('nOut_pool_h', nOut_pool_h)
        print('nOut_pool_w', nOut_pool_w)
        
        print(nOutputs_last)
        --]]
        --- C3 ; S4 ------


        local reshape_module = nn.Reshape(nOutputs_last)
        if not useCUDAmodules then
            reshape_module.vector = true
        end
        feat_extractor:add(reshape_module)

        local nStatesFC = networkOpts.nStatesFC
        local nFClayers = #nStatesFC

        nUnitsInLastLayer = nOutputs_last
        -- fully-connected layers (if any)
        if nFClayers > 0 then
            for layer_i,nUnitsInThisLayer in ipairs(nStatesFC) do
                feat_extractor:add(  nn.Linear(nUnitsInLastLayer, nUnitsInThisLayer) )
                feat_extractor:add(  nn.Tanh() )
                
                nUnitsInLastLayer = nUnitsInThisLayer
            end
        end
 
  
        --classifier 
        classifier:add(  nn.Linear(nUnitsInLastLayer, nClasses) )
        classifier:add(  nn.LogSoftMax() )
        
        if nPositions > 1 then
            local indiv_pos_classifier = nn.Linear(nUnitsInLastLayer, nClasses * nPositions);
            
            classifier_indiv_pos:add( indiv_pos_classifier )  -- classifier for each individual position
            classifier_indiv_pos:add( nn.LogSoftMax() )
            
            classifier_indiv_pos_combined:add(  indiv_pos_classifier )  -- classifier for each letter, taking in inputs from each letter at each individual position
            classifier_indiv_pos_combined:add(  nn.Tanh() )
            classifier_indiv_pos_combined:add(  nn.Linear(nClasses * nPositions, nClasses) )
            classifier_indiv_pos_combined:add(  nn.LogSoftMax() )
        end
 
 
        params.nStatesConv = nStatesConv
        params.nStatesFC = nStatesFC
        params.nConvLayers = nConvLayers
        params.fanin = fanin
        params.filtSizes = filtSizes
        params.poolSizes = poolSizes
        params.poolStrides = poolStrides
        params.poolTypes = poolTypes
        params.nOutputs_last = nOutputs_last
        params.connectTables = connectTables 
        --params.useConnectionTable = useConnectionTable
        params.convFunction = networkOpts.convFunction
        params.trainOnGPU = trainOnGPU
        
        params.resizeInputToVector = false

    else 
        error('Unknown network type')
    end
    


    -- add model (with feature extractor & classifier) to the model_struct container
    local model_struct = {}
    model_struct.parameters = params
     
    
    local model = nn.Sequential()
    local model_indiv_pos, model_indiv_pos_combined
    
    model:add(feat_extractor)
    model:add(classifier)
   
    --model = feat_extractor
   
    model_struct.model = model   
    model_struct.model_combined_pos = model   
    
    if nPositions > 1 then
        model_indiv_pos = nn.Sequential()
        model_indiv_pos:add(feat_extractor)
        model_indiv_pos:add(classifier_indiv_pos)
        model_struct.model_indiv_pos = model_indiv_pos
        
        
        model_indiv_pos_combined = nn.Sequential()
        model_indiv_pos_combined:add(feat_extractor)
        model_indiv_pos_combined:add(classifier_indiv_pos_combined)
        model_struct.model_indiv_pos_combined = model_indiv_pos_combined
    end
        
        --model_struct.feat_extractor = feat_extractor
        --model_struct.classifier = classifier      

    
    
    --[[
    model_struct.gpu = {}
    model_struct.gpu.model          = model_gpu
    model_struct.gpu.feat_extractor = feat_extractor_gpu
    model_struct.gpu.classifier     = classifier_gpu
    --]]
    model_struct.modelType = 'model_struct'    -- to indicate to functions that this is 
                                               -- just a container for the model, not the model itself             
    model_struct.criterion =  nn.ClassNLLCriterion()
      
    if networkOpts.trainOnGPU then
        moveModelToGPU(model_struct)
    end
      
    --model_struct.reshapeImage = reshapeImage
    return model_struct
    
end



moveModelToGPU = function(model_struct)

    --local cur_model_use = model_struct.model
    MS = model_struct
    
    if not (model_struct.modelIsOnGPU == true) then  -- ie. model is on CPU
    
        -- check whether the model has SpatialConvolutionMap (which is not currently compatible with the GPU)
        
        if model_struct.parameters.useConnectionTable or areAnySubModulesOfType(model_struct.model, 'nn.SpatialConvolutionMap') then
            error('Cannot move model to the GPU : model uses the "SpatialConvolutionMap" module')
        end
    
        
        if not model_struct.model_onGPU then
            io.write('Moving Model to the GPU for the first time ... ')
            
            model_struct.model_onCPU = model_struct.model
            
            local model_onGPU = nn.Sequential()
            model_onGPU:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
            model_onGPU:add(model_struct.model_onCPU:cuda())
            model_onGPU:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
            model_struct.model_onGPU = model_onGPU
            
        else
            io.write('Moving model back to the GPU ... ')
            model_struct.model_onGPU.modules[2]:cuda() -- copy back to the GPU
            
        end
        
        io.write(' done.\n')
        
        model_struct.model = model_struct.model_onGPU
            
        model_struct.modelIsOnGPU = true
    end
        
    
end

moveModelBackToCPU = function(model_struct)

    --local cur_model_use = model_struct.model
    if (model_struct.modelIsOnGPU == true) then
        io.write('Moving model back to the CPU ... \n')
        model_struct.model = model_struct.model_onCPU:float()   -- this is the original model (without the CudaTensor <-> FloatTensor copying).
    
        model_struct.modelIsOnGPU = false
    end
        
    
end


getNormKernel = function(kernel_type, kernel_width)
   assert(kernel_type ~= 'none')
   if string.lower(kernel_type) == 'gauss' then
        return image.gaussian1D(kernel_width)
   else
        error('Unknown kernel type : ' .. kernel_type)
   end
   
    
end

areAnySubModulesOfType = function(curModule, nn_type)
    
    
    if curModule.modules then
        for i,mod in ipairs(curModule.modules) do
            --io.write(string.format('%d, %s\n', i, torch.typename(mod) ))
            
            if torch.typename(mod) == nn_type then
                --io.write('Equal!')
                return true
            end
            
            if torch.typename(mod) == 'nn.Sequential' then
                if areAnySubModulesOfType(mod, nn_type) then
                    return true
                end
            end 
            
        end
    end
    
    return false
end

networkHash = function(model, includeOutputFields, includeGradFields)
    
    local hash_fields = {'bias', 'weight'}
    if includeOutputFields then
        hash_fields = table.merge(hash_fields, {'output'})
    end
    if includeGradFields then
        hash_fields = table.merge(hash_fields, {'gradBias', 'gradWeight', 'gradInput'})
    end
    
    local hash = 0
    local level = 1
    local debug = false
    
    local function recurseForModule(mod, lvl)        
            
        for i,fld in ipairs(hash_fields) do
            if mod[fld] then
                local hash_thisField = torch.abs(mod[fld]):sum()
                hash = hash + hash_thisField
                if debug then
                    for j = 1,lvl do
                        io.write('  ')
                    end
                    io.write(string.format("adding hash for %s (%.4f), size = %s\n", fld, hash_thisField, toList(mod[fld]:size(), 'x')))
                end
            end                    
        end
                            
        if mod.modules then
                                
            for submod_i = 1, #mod.modules do
                if debug then
                    io.write(string.format('%d . %d\n', lvl, submod_i))
                end
                recurseForModule(mod.modules[submod_i], lvl + 1)
            end
            
        end
        
        
    end
            
    recurseForModule(model, 1)   
      
    return hash
        
    
end


getDefaultConvNetParams = function()
        
    local params = {}
    params.nStatesConv = {6,16}
    params.nStatesFC = {120}    
    params.convFunction = 'SpatialConvolutionMap'
    
    params.fanin = {1,4,16}
    params.filtSizes = {5,4}

    params.doPooling = true
    params.poolSizes = {4,2}
    params.poolTypes = {2,2}
    params.poolStrides = 'auto' --{2,2}

    params.doSpatSubtrNorm = false
    params.doSpatDivNorm = false    

    return params
    

end

--[[
getStreamlinedModel = function(model)
    
    if (#model.modules == 2) 
            and (string.sub( tostring(model.modules[1]), 1, 13) == 'nn.Sequential') 
            and (string.sub( tostring(model.modules[2]), 1, 13) == 'nn.Sequential')  then
        
        local full_model = nn.Sequential()
        for idx_mod1 = 1,#model.modules[1].modules do
            full_model:add ( model.modules[1].modules[idx_mod1] )         
        end
        for idx_mod2 = 1,#model.modules[2].modules do
            full_model:add ( model.modules[2].modules[idx_mod2] )         
        end
        return full_model
    else
        return model
    end
    
end
--]]

getStreamlinedModel = function(model)
    
    local full_model = nn.Sequential()
    if not model.modules then
        full_model:add(model)
        return
    end
        
    local nModules = #model.modules
    
    for i, mod in ipairs(model.modules) do
        local module_type = torch.typename(mod)            
        if module_type == 'nn.Sequential' then
            local subModel = getStreamlinedModel(mod)
            ---[[
            for j, subModule in ipairs(subModel.modules) do
                full_model:add(subModule)
            end
            --]]
        else        --if module_type == 'nn.SpatialLPPooling' then
            full_model:add(mod)
            
        end
    end
    
    return full_model
end
   --[[     
    if (#model.modules == 2) 
            and (string.sub( tostring(model.modules[1]), 1, 13) == 'nn.Sequential') 
            and (string.sub( tostring(model.modules[2]), 1, 13) == 'nn.Sequential')  then
        
        local full_model = nn.Sequential()
        for idx_mod1 = 1,#model.modules[1].modules do
            full_model:add ( model.modules[1].modules[idx_mod1] )         
        end
        for idx_mod2 = 1,#model.modules[2].modules do
            full_model:add ( model.modules[2].modules[idx_mod2] )         
        end
        return full_model
    else
        return model
    end
    
end
--]]


getModelSequence = function(model)
    local fullModel = getStreamlinedModel(model)
    local seq = {}
    local nModules = #fullModel.modules
    for idx = 1,nModules do
        table.insert(seq, torch.typename(fullModel.modules[idx]))
    end
        
    return seq
end


getModuleIndex = function(model, moduleToFind_full, errorIfNotFound)
    
    local moduleToFind, occurence = stringAndNumber(moduleToFind_full)
    occurence = occurence  or 1
    
    moduleToFind = string.lower ( string.gsub(moduleToFind, 'nn.', '') )
    
    local seq = getModelSequence(model)
    local nModules = #seq
    local module_idx
    
    local parseOccurence = function(occurence, n)
        --[[
        if occurence == 'first' then
            occurence = 1
        end 
        if occurence == 'last' then
            occurence = -1
        end
        --]]
        if occurence < 0 then
            occurence = n + occurence + 1
        end
        return occurence
    end

    local isConvStr = function(str) return string.find(string.lower(str), 'conv') ~= nil; end
    
    local moduleToFind_orig = moduleToFind
    if moduleToFind == 'classifier' then
        moduleToFind = 'linear'
        occurence = -1
    end

        
    if isConvStr(moduleToFind) then
        --print('Searching for conv...')
        local conv_indices = table.find(seq, isConvStr, 'all')
        --print(conv_indices)
        occurence = parseOccurence(occurence, #conv_indices)        
        
        module_idx = conv_indices[occurence]
                    
    else 
        local module_str = 'nn.' .. string.titleCase(moduleToFind)
                
        local linear_indices = table.find(seq, module_str, 'all')
        if #linear_indices == 0 then
            if errorIfNotFound then
                error(string.format('Could not find any modules of type %s ("%s") in the model\n', moduleToFind, module_str))
            else
                return nil
            end
                
        end
        occurence = parseOccurence(occurence, #linear_indices)
        
        module_idx = linear_indices[occurence]
        if not module_idx then
            if errorIfNotFound then
                error(string.format('There are only %d occurences of %s (could not access # %d)\n', #linear_indices, module_str, occurence))
            else
                return nil
            end
        end
        
        if moduleToFind_orig == 'classifier' then
            assert(seq[module_idx] == 'nn.Linear')
            assert(seq[module_idx+1] == 'nn.LogSoftMax')
        end
        
    end

    return module_idx
end



copyConvolutionalFiltersToNetwork = function (model_struct1, model_struct2)
    
    local full_model1 = getStreamlinedModel(model_struct1.model)
    local full_model2 = getStreamlinedModel(model_struct2.model)
    F1 = full_model1
    F2 = full_model2
    
    
    
    local net1_idx, net2_idx, convLayer_idx
    convLayer_idx = 1
    repeat 
        net1_idx = getModuleIndex(full_model1, 'Conv' .. convLayer_idx)
        net2_idx = getModuleIndex(full_model2, 'Conv' .. convLayer_idx)
        if net1_idx and net2_idx then
            local w1 = full_model1.modules[net1_idx].weight
            local w2 = full_model2.modules[net2_idx].weight
            
            io.write(string.format('Copying Conv filter #%d from network1 (size=%s) to network2 (size=%s) \n', 
                    convLayer_idx, tostring_inline(w1:size()), tostring_inline(w2:size()) ) )
            --copyConvolutionalWeights(full_model1, net1_idx, full_model2, net2_idx)
            w1:copy(w2)
            
            --full_model1.modules[net1_idx].weight:copy(full_model2.modules[net2_idx].weight)
    
        elseif net1_idx and not net2_idx then
            print('Network 1 has a Conv filter #' .. convLayer_idx .. ', but network #2 does not...')
        
        elseif not net1_idx and net2_idx then
            print('Network 1 does not have a Conv filter #' .. convLayer_idx .. ', but network #2 does...')
        
        end    
    
        convLayer_idx = convLayer_idx + 1
        
    until (not net1_idx and not net2_idx)
    
   
    
end

copyConvolutionalWeights = function (model1, mod1_idx, model2, mod2_idx)
    M1 = model1
    i1 = mod1_idx
    M2 = model2
    i2 = mod2_idx
    error('!')
   
    
   
   
end



resizeConvolutionalOutputLayers = function(model_struct, retrainImageSize)
    
    local full_model = getStreamlinedModel(model_struct.model)
    F = full_model
    
    
    
end


replaceClassifierLayer = function(model_struct, nClasses)

    local full_model = getStreamlinedModel(model_struct.model)
    
    local classifier_idx = getModuleIndex(full_model, 'classifier')
    
    local nModules = #(full_model.modules)
    
    print(string.format('Replacing the classifier with a new one with %d classes', nClasses))
    
    local model_new = nn.Sequential()
    
    for layer_i, curModule in ipairs(full_model.modules) do
        
        if layer_i == classifier_idx then
                        
            assert(torch.typename(curModule) == 'nn.Linear')
            local nClassesPrev = curModule.weight:size(1)
            local nUnitsInLastLayer = curModule.weight:size(2)
            
            --local nUnitsInLastLayer = prevModule.output:size()[1]
            local newClassifierModule = nn.Linear(nUnitsInLastLayer, nClasses)
            if curModule.bias:type() == 'torch.CudaTensor' then
                newClassifierModule:cuda() 
            end
            io.write(string.format('  [Old classifier was %d => %d. New classifier is %d => %d]\n',  
                    nUnitsInLastLayer, nClassesPrev,   nUnitsInLastLayer, nClasses )) 
            
            model_new:add( newClassifierModule  )   -- new classifier layer
        else
            model_new:add( curModule )        
        end
                
        if layer_i == classifier_idx+1 then
            -- make sure layer after the classifier is a logsoftmax module
            assert(torch.typename(curModule) == 'nn.LogSoftMax')
        end

    end
    
    model_struct.model = model_new
    
    return model_struct
    
    
end


splitModelAfterLayer = function(model_struct, splitLayer)
    
    local full_model = getStreamlinedModel(model_struct.model)
    local addCopyUnitsForGPUmodel = true
    --local modelSequence = getModelSequence(model_struct.model)
    local splitLayer_str
    if type(splitLayer) == 'string' then
        splitLayer_str = splitLayer
        splitLayer = getModuleIndex(full_model, splitLayer)
    end    
    assert(type(splitLayer) == 'number')
    splitLayer_str = splitLayer_str or string.format('Layer %d', splitLayer)
    
    local nModules = #(full_model.modules)
    if splitLayer < 0 then
        assert( splitLayer ~= -1) -- this would be splitting *after* the last layer, ie. not splitting at all.
        splitLayer = nModules - (-splitLayer) + 1   --     -1 = last layer
    end
    print(string.format('\nSplitting the network from %s (layer #%d)\n', splitLayer_str, splitLayer))
    
    local feat_extractor = nn.Sequential() --- before split
    local classifier = nn.Sequential()  --- after split
    
    io.write('Splitting model : \n * Feature extractor = \n    ')
    
    local startedClassifier = false
    local modelOnGPU
    for layer_i = 1, nModules do
        local curModule = full_model.modules[layer_i]
        
        if layer_i == 1  then
            modelOnGPU = torch.typename(curModule) == 'nn.Copy'
        end
                    
        if layer_i < splitLayer then   -- 6/15 changed < to <=.   (redo retrained models done before this date).. 7/30 changed back to <
            feat_extractor:add( curModule  )
        else
            if not startedClassifier then
                
                if modelOnGPU and addCopyUnitsForGPUmodel then
                     feat_extractor:add( nn.Copy('torch.CudaTensor', 'torch.FloatTensor' ) )
                     io.write('[+nn.Copy]; ') 
                end
                
                io.write('\n * Classifier = \n    ') 
                if modelOnGPU and addCopyUnitsForGPUmodel then
                    classifier:add( nn.Copy('torch.FloatTensor', 'torch.CudaTensor') )
                    io.write('[+nn.Copy]; ') 
                end
                
                startedClassifier = true
            end
            classifier:add( curModule  )
        end
        
        if layer_i == nModules and modelOnGPU then
            assert(torch.typename(curModule) == 'nn.Copy')
        end
        
        io.write(string.format('%s; ', torch.typename(curModule) ))
        
    end
    print('')
    
    model_struct.feat_extractor = feat_extractor
    model_struct.classifier = classifier
    return model_struct
    
    
end


visualizeHiddenUnits = function(model)
    
    local wgtMtx = model.modules[2].weight
    local nUnits = wgtMtx:size(1)
    local nPix = wgtMtx:size(2)
    local nPixSide = math.sqrt(nPix)
    
    local nRows = math.floor(math.sqrt(nUnits))
    local nCols = math.ceil(nUnits / nRows)
    
    local displayMtx = torch.Tensor(nRows * (nPixSide+1) + 2, nCols * (nPixSide+1) + 2):zero()
    local u_idx = 1
    local mn_val = torch.min(wgtMtx)
    local mx_val = torch.max(wgtMtx)
    local wgt_i
    
    for R = 1,nRows do
        for C = 1,nCols do
            if u_idx > nUnits then
                break
            end
            
            
            wgt_i = rescale( wgtMtx[u_idx]:resize(nPixSide, nPixSide), mn_val, mx_val)
            
            for r = 1,nPixSide do
                for c = 1,nPixSide do
                    displayMtx[ 1 + (R-1)*(nPixSide+1) + r][ 1+ (C-1)*(nPixSide+1) + c] = wgt_i[r][c]
                end
            end
            
            u_idx = u_idx + 1
            
        end
    end
    
    image.display(displayMtx)
    
    
end

rescale = function(X, min_val, max_val)
    return (X-min_val)/(max_val-min_val)
    
end

    
copyModelUnits = function(model_struct1, model_struct2)
    
    feat_extractor1 = model_struct1.feat_extractor    
    feat_extractor2 = model_struct2.feat_extractor    
    module_left_lin1 = feat_extractor1.modules[2]
    module_left_lin2 = feat_extractor2.modules[2]

    classifier1 = model_struct1.classifier
    classifier2 = model_struct2.classifier
    module_right1 = classifier1.modules[1]
    module_right2 = classifier2.modules[1]
    
    nOutputs1 = module_left_lin1.output:numel()
    nOutputs2 = module_left_lin2.output:numel()
    assert(nOutputs1 == module_right1.weight:size(2))
    assert(nOutputs2 == module_right2.weight:size(2))
    
    nInputsToCopy = math.min(nOutputs1, nOutputs2)
    idx_cpy = {1, nInputsToCopy}
        
    --module_left.bias = module_left.bias[idx_remain]
    --feat_extractor.output = feat_extractor.output[{ {}, {1, nInputsRemain} }]

    module_left_lin2.bias       = module_left_lin1.bias[{{1, nInputsToCopy}}]:clone()
    --module_left_lin2.gradBias   = module_left_lin1.gradBias[{{1, nInputsToCopy}}]
    module_left_lin2.weight     = module_left_lin1.weight[{ {1, nInputsToCopy}, {} }]:clone()
    --module_left_lin2.gradWeight = module_left_lin1.gradWeight[{ {1, nInputsToCopy}, {} }]
    --module_left_lin2.output     = module_left_lin1.output[{{},  {1, nInputsToCopy}}]
    --module_left_tanh2.gradInput = module_left_tanh1.gradInput[{{1, nInputsToCopy}}]
    --module_left_tanh2.output    = module_left_tanh1.output[{{},  {1, nInputsToCopy}}]
        
    --classifier2.output       = classifier1.output[{{},  {1, nInputsToCopy}}]
    --classifier2.gradInput    = classifier1.gradInput[{{1, nInputsToCopy}}]
    module_right2.weight     = module_right1.weight[{ {}, {1, nInputsToCopy} }]:clone()
    --module_right2.gradWeight = module_right1.gradWeight[{ {}, {1, nInputsToCopy} }] 
    --module_right2.gradInput  = module_right1.gradInput[{{1, nInputsToCopy}}]
    
    
end
    
    
--[[    
function Module:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:type(type)
      end
   end
   return self
end
--]]
        
        


fixTopLayer = function ( topModule, nClasses, nPositions )
    
    local bias = topModule.bias
    local weight = topModule.weight
    
    --print(bias:size(1), nClasses)
    assert(bias:size(1) == nClasses)
    assert(weight:size(1) == nClasses)
    assert(weight:size(2) == nClasses*nPositions)
    for i = 1,nClasses do
        bias[i] = 0
        
        for j = 1,nClasses * nPositions do
            if math.ceil(j/nPositions) == i   then
                weight[i][j] = 0.5
            else
                weight[i][j] = 0
            end
        end
       
    end
    
    
end


convertNetworkToMatlabFormat = function(model)

    local net = {}

    --net.network_str = ''
    net.modules_str = ''
    net.nModules = 1

    addModules = function(net, modules)
        --print('-----');
        --print(modules)
        --print('length = ', #modules)
        
        for mod_j, module_i in ipairs(modules) do
            
            local module_full_str = tostring(module_i)
            --print(module_full_str )
            assert(string.sub(module_full_str, 1, 3) == 'nn.')
            
            local module_str = string.sub(module_full_str, 4, #module_full_str)
            
            local j = net.nModules
            
            if string.sub(module_str, 1, 10) == 'Sequential' then
--                net.networkStr = net.networkStr .. module_full_str .. '
                
                net = addModules(net, module_i.modules)
        
            else 
                net[ 'm' .. j .. '_str'] = str2vec(module_str)
                local module_name_str = module_str
                
                if module_str == 'SpatialConvolutionMap' or module_str == 'SpatialConvolution' or  module_str == 'SpatialConvolutionCUDA'  then
                    
                    net[ 'm' .. j .. '_bias'] = module_i.bias:double()
                    net[ 'm' .. j .. '_weight'] = module_i.weight:double()
                    net[ 'm' .. j .. '_nInputPlane'] = torch.DoubleTensor({module_i.nInputPlane})
                    net[ 'm' .. j .. '_nOutputPlane'] = torch.DoubleTensor({module_i.nOutputPlane})
                    net[ 'm' .. j .. '_kH'] = torch.DoubleTensor({module_i.kH})
                    net[ 'm' .. j .. '_kW'] = torch.DoubleTensor({module_i.kW})
                    net[ 'm' .. j .. '_dH'] = torch.DoubleTensor({module_i.dH})
                    net[ 'm' .. j .. '_dW'] = torch.DoubleTensor({module_i.dW})
                    if module_i.connTable then
                        net[ 'm' .. j .. '_connTable'] = module_i.connTable:double()
                    end
                    
                    module_name_str = 'Conv'
                                
                elseif (module_str == 'SpatialSubSampling') then                    
                    net[ 'm' .. j .. '_kH'] = torch.DoubleTensor({module_i.kH})
                    net[ 'm' .. j .. '_kW'] = torch.DoubleTensor({module_i.kW})
                    net[ 'm' .. j .. '_dH'] = torch.DoubleTensor({module_i.dH})
                    net[ 'm' .. j .. '_dW'] = torch.DoubleTensor({module_i.dW})
                    net[ 'm' .. j .. '_connTable'] = module_i.connTable
                    
                    module_name_str = 'SubSamp'
                    
                elseif (module_str == 'SpatialMaxPooling') or (module_str == 'SpatialMaxPoolingCUDA') then                    
                    net[ 'm' .. j .. '_kH'] = torch.DoubleTensor({module_i.kH})
                    net[ 'm' .. j .. '_kW'] = torch.DoubleTensor({module_i.kW})
                    net[ 'm' .. j .. '_dH'] = torch.DoubleTensor({module_i.dH})
                    net[ 'm' .. j .. '_dW'] = torch.DoubleTensor({module_i.dW})
                    if module_i.indices then
                        net[ 'm' .. j .. '_indices'] = module_i.indices:double()
                    end
                    
                    module_name_str = 'MaxPool'
                                        
                elseif module_str == 'Linear' then
                    
                    net[ 'm' .. j .. '_bias'] = module_i.bias:double()
                    net[ 'm' .. j .. '_weight'] = module_i.weight:double()
                
                elseif module_str == 'Transpose' then
                    
                    net[ 'm' .. j .. '_permutations'] = tbltbl2Tensor ( module_i.permutations ):double()
                                
                elseif (module_str == 'Square') or (module_str == 'Sqrt') or (module_str == 'Copy') or 
                    (module_str == 'Tanh') or (module_str == 'Reshape') or (module_str == 'LogSoftMax') or (module_str == 'Exp')  then     
                
                else
                    error('Unhandled case : module type = ' .. module_str)
                
                end
                                
                net.modules_str = net.modules_str .. module_name_str .. ';' 
                net.nModules = net.nModules + 1

            end
        end    
        
        return net
    end

    net = addModules(net, model.modules)
    
    net.nModules = torch.DoubleTensor({net.nModules - 1})
    net.modules_str = str2vec(net.modules_str)
    return net
    
end

str2vec = function(s)
    v = torch.DoubleTensor(#s)
    for i = 1,#s do
        v[i] = string.byte(s,i,i)    
    end
    return v
end


nOutputsFromConvStages = function(networkOpts, imageSize)
    
    networkOpts = fixConvNetParams(networkOpts)
    --print(networkOpts)
        
    local height, width = imageSize[1], imageSize[2]
    
    
    local nStatesConv = networkOpts.nStatesConv
    nStatesConv[0] = nInputPlanes
    local nConvLayers = #nStatesConv
    
    local filtSizes,          doPooling,              poolSizes,             poolTypes,              poolStrides = 
        networkOpts.filtSizes, networkOpts.doPooling, networkOpts.poolSizes, networkOpts.poolTypes,  networkOpts.poolStrides
            
    --print('poolTypes = ', poolTypes)
    NetworkOpts = networkOpts
        
    local nOut_conv_h = {}
    local nOut_conv_w = {}
    local nOut_pool_h = {}
    local nOut_pool_w = {}
    nOut_pool_h[0] = height
    nOut_pool_w[0] = width
        
    for layer_i = 1,nConvLayers do
        -- 1. Convolutional layer
        
        if (filtSizes[layer_i] > 0) then -- 
            
            nOut_conv_h[layer_i] = rectified( nOut_pool_h[layer_i-1] - filtSizes[layer_i] + 1 )
            nOut_conv_w[layer_i] = rectified( nOut_pool_w[layer_i-1] - filtSizes[layer_i] + 1 )
            
        elseif (filtSizes[layer_i] == 0) then   -- use filtSize == 0, as flag to skip convolution step (& tanh) in this layer
            nStatesConv[layer_i] = nStatesConv[layer_i-1]
            
            nOut_conv_h[layer_i] = nOut_pool_h[layer_i-1]
            nOut_conv_w[layer_i] = nOut_pool_w[layer_i-1]
        end
            
        -- 3. Spatial pooling / sub-sampling
        local poolType_thisLayer = poolTypes[layer_i]
        if doPooling and not (poolType_thisLayer == 0) and (nOut_conv_h[layer_i] > 0) and  (nOut_conv_w[layer_i] > 0) then
            assert(poolStrides[layer_i] <= poolSizes[layer_i])
            
            nOut_pool_h[layer_i] = math.floor( (nOut_conv_h[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
            nOut_pool_w[layer_i] = math.floor( (nOut_conv_w[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
        else
            nOut_pool_h[layer_i] = nOut_conv_h[layer_i]
            nOut_pool_w[layer_i] = nOut_conv_w[layer_i]            
        end
    end
            
    local nUnitsInLastLayerPerBank = (nOut_pool_h[nConvLayers]*nOut_pool_w[nConvLayers])
    local nOutputs_last = nStatesConv[nConvLayers] * nUnitsInLastLayerPerBank
        
    return nOutputs_last, nUnitsInLastLayerPerBank
end
        