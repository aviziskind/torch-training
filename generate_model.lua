require 'image'  -- for normalization kernels

generateModel = function(inputStats, networkOpts, letterOpts)
    
    --torch.manualSeed(123)
    local nInputs = inputStats.nInputs
    local nInputPlanes = inputStats.nInputPlanes or 1
    local height  = inputStats.height;        assert(height)
    local width   = inputStats.width;         assert(width)
    local nOutputs = inputStats.nClasses or inputStats.nOutputs;  assert(nOutputs)
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
        nOutputs = fontClassesTable.nOutputsTot
    end

    local params = {}

    local finalLayer = networkOpts.finalLayer
    assert(finalLayer)

    local nLinType = networkOpts.nLinType or 'Tanh'
    assert(nLinType)

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
            
            local nlin = getNonlinearity(nLinType) 
            feat_extractor:add(  nlin )

            nUnitsInLastLayer = nUnitsInThisLayer
        end

        --output layer
        classifier:add(  nn.Linear(nUnitsInLastLayer, nOutputs) )

        if finalLayer == 'LogSoftMax' then
            classifier:add(  nn.LogSoftMax() )
        elseif finalLayer == '' then
            -- do nothing 
        else
            error('Unknown final layer type : ' .. finalLayer )
        end

        if nPositions > 1 then
            local indiv_pos_classifier = nn.Linear(nUnitsInLastLayer, nOutputs * nPositions);

            classifier_indiv_pos:add( indiv_pos_classifier )  -- classifier for each individual position
            classifier_indiv_pos:add( nn.LogSoftMax() )

            classifier_indiv_pos_combined:add(  indiv_pos_classifier )  -- classifier for each letter, taking in inputs from each letter at each individual position
            classifier_indiv_pos_combined:add(  nn.LogSoftMax() )
            classifier_indiv_pos_combined:add(  nn.Exp() )
            classifier_indiv_pos_combined:add(  nn.Linear(nOutputs * nPositions, nOutputs) )
            classifier_indiv_pos_combined:add(  nn.LogSoftMax() )
        end

        --classifier:add(  nn.Reshape(nOutputs) )
        params.nHiddenUnitsEachLayer = nHiddenUnitsEachLayer
        params.resizeInputToVector = true
        --reshapeImage = true

    elseif networkOpts.netType == 'ConvNet' then

        local defaultNetworkName = networkOpts.defaultNet
        networkOpts = fixConvNetParams(networkOpts, defaultNetworkName)
        --NetworkOpts = networkOpts

        local nInputPlanes = nInputPlanes or 1 

        local nStatesConv = networkOpts.nStatesConv
        local nStatesFC   = networkOpts.nStatesFC
        nStatesConv[0] = nInputPlanes
        local nConvLayers = #nStatesConv
        local nFCLayers = #nStatesFC
        --nStates_copy2 = nStatesConv

        --print(nStatesConv)
        --local useConnectionTable_default = true
        --local useConnectionTable_default = false
        --local useConnectionTable = useConnectionTable_default and not trainOnGPU
        --params.enforceStridePoolSizeEqual = trainOnGPU
        local n = networkOpts
        
        local convFunction,   fanin,   filtSizes,   doPooling,   poolSizes,   poolTypes,   poolStrides,   trainOnGPU = 
            n.convFunction, n.fanin, n.filtSizes, n.doPooling, n.poolSizes, n.poolTypes, n.poolStrides, n.trainOnGPU
        
        local doSpatSubtrNorm,   spatSubtrNormType,   spatSubtrNormWidth,   doSpatDivNorm,   spatDivNormType,   spatDivNormWidth = 
            n.doSpatSubtrNorm, n.spatSubtrNormType, n.spatSubtrNormWidth, n.doSpatDivNorm, n.spatDivNormType, n.spatDivNormWidth
           

        local zeroPadForPooling = n.zeroPadForPooling or true
        local zeroPadForConvolutions = n.zeroPadForConvolutions or false
        local splitInTwo = function(x)
            local a = math.ceil(x/2)
            local b = x-a 
            return a, b
        end

        local convPs, fcPs
        local convLayerDropoutPs = table.rep(0, nConvLayers)
        local fullyConnectedDropoutPs = table.rep(0, nFCLayers)
        
        local dropoutPs = n.dropoutPs
        local useSpatialDropout = n.spatialDropout
        if dropoutPs then
            if type(dropoutPs) == 'number' then     
                if dropoutPs > 0 then        -- dropoutPs = 0.5 : dropout after final convolutional layer with p = 0.5
                    convPs = {dropoutPs}
                elseif dropoutPs < 0 then    -- dropoutPs = -0.5 : dropout in all fully connected layers with p = 0.5
                    fcPs = {-dropoutPs} --fullyConnectedDropoutPs = table.rep(-dropoutPs, nFCLayers)
                end
                    
                    
            elseif type(dropoutPs) == 'table' then
                local idx_firstPos = table.find(dropoutPs, function (x) return x>0; end)
                local idx_firstNeg = table.find(dropoutPs, function (x) return x<0; end)
                local convPs, fcPs
                if idx_firstPos ~= nil then
                    if idx_firstNeg ~= nil then
                        convPs, fcPs = table.split(dropoutPs, idx_firstNeg)
                    else
                        convPs = dropoutPs
                    end
                elseif idx_firstNeg then
                    fcPs = dropoutPs
                end
            end
                
            if convPs then
                assert(#convPs <= nConvLayers)
                for j = 1,#convPs do
                    convLayerDropoutPs[nConvLayers - #convPs + j] =  convPs[j]
                end
            end
                
            if fcPs then    -- dropoutPs = {0.5, -0.5} : dropout in all convolutional layers with p = 0.5
                fullyConnectedDropoutPs = table.apply(fcPs, function(x) return -x; end)  -- extend
                if #fcPs == nFCLayers then
                    fullyConnectedDropoutPs = fcPs  
                else
                    assert(#fcPs == 1)
                    fullyConnectedDropoutPs = table.rep(-fcPs[1], nFCLayers)
                end
                
            end
            --print(convPs)
            --print('convLayerDropoutPs', convLayerDropoutPs)
            --print('fullyConnectedDropoutPs', fullyConnectedDropoutPs)
                            
        end

        local doDropoutInConvLayers = #convLayerDropoutPs > 0
        local doDropoutInFullyConnectedLayers = #fullyConnectedDropoutPs > 0

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
        local nOut_pool_h,  nOut_pool_h_uncropped = {}
        local nOut_pool_w,  nOut_pool_w_uncropped = {}
        nOut_pool_h[0] = height
        nOut_pool_w[0] = width


        if useCUDAmodules then
            feat_extractor:add(nn.Transpose({1,4},{1,3},{1,2}))  -- transpose so that batch dim is last (CUDA modules expect nPlanes x h x w x batchSize)
        end        

        for layer_i = 1,nConvLayers do
            -- 1. Convolutional layer
            local SpatConvModule
            local kW, kH = filtSizes[layer_i], filtSizes[layer_i]
            local dW, dH = 1, 1
            if (filtSizes[layer_i] > 0) then 

                local nConvPaddingW, nConvPaddingH = 0,0
                if zeroPadForConvolutions then
                    if type(zeroPadForConvolutions) == 'number' then
                        nConvPaddingW = zeroPadForConvolutions
                        nConvPaddingH = nConvPaddingW
                    elseif type(zeroPadForConvolutions) == 'boolean' and zeroPadForConvolutions == true then
                        nConvPaddingW = math.floor( (kW-1)/2 )
                        nConvPaddingH = math.floor( (kH-1)/2 )
                    end
                end

                if convFunction == 'SpatialConvolutionMap' then
                    -- fanin: how many incoming connections (from a state) in previous layer each output unit receives input from.
                    -- this cant be more than the number of states in previous layer!
                    assert(dW == 1 and dH == 1)
                    Fanin = fanin
                    NStatesConv = nStatesConv
                    Layer_i = layer_i
                    local fanin_use = math.min(fanin[layer_i], nStatesConv[layer_i-1]) 
                    connectTables[layer_i] = nn.tables.random(nStatesConv[layer_i-1], nStatesConv[layer_i], fanin_use) -- OK b/c only 1 feature--- 
                    SpatConvModule = nn.SpatialConvolutionMap(connectTables[layer_i], kW, kH)

                    if nConvPaddingW > 0 or nConvPaddingH > 0 then
                        local convPaddingModule = nn.SpatialZeroPadding(nConvPaddingW, nConvPaddingW, nConvPaddingH, nConvPaddingH)
                        feat_extractor:add(convPaddingModule)
                    end

                elseif convFunction == 'SpatialConvolution' then
                    SpatConvModule = nn.SpatialConvolution(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH,  dW, dH,  nConvPaddingW, nConvPaddingH)

                --elseif convFunction == 'SpatialConvolutionCUDA' then   (deprecated)
                  --  SpatConvModule = nn.SpatialConvolutionCUDA(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH)

                elseif convFunction == 'SpatialConvolutionMM' then
                    SpatConvModule = nn.SpatialConvolutionMM(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH, dW, dH,  nConvPaddingW, nConvPaddingH)
                
                elseif convFunction == 'SpatialConvolutionCUDNN' then
                    SpatConvModule = cudnn.SpatialConvolution(nStatesConv[layer_i-1], nStatesConv[layer_i], kW, kH,  dW, dH,  nConvPaddingW, nConvPaddingH)
                    
                else
                    error('Unknown spatial convolution function : ' .. tostring(convFunction))
                end
                feat_extractor:add(SpatConvModule)

                
                nOut_conv_w[layer_i] = nOut_pool_w[layer_i-1] - filtSizes[layer_i] + 1  + nConvPaddingW*2
                nOut_conv_h[layer_i] = nOut_pool_h[layer_i-1] - filtSizes[layer_i] + 1  + nConvPaddingH*2

                -- 2. Nonlinearity (sigmoid)
                local nlin = getNonlinearity(nLinType) 
                feat_extractor:add( nlin )


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

                if useCUDAmodules and convFunction == 'SpatialConvolutionCUDA' then
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

                nOut_pool_h[layer_i] = math.floor( (nOut_conv_h[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
                nOut_pool_w[layer_i] = math.floor( (nOut_conv_w[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1

                local nCovered_pool_h = (nOut_pool_h[layer_i] - 1) * poolStrides[layer_i] + poolSizes[layer_i]
                local nCovered_pool_w = (nOut_pool_w[layer_i] - 1) * poolStrides[layer_i] + poolSizes[layer_i]

                local dropped_pixels_h = nOut_conv_h[layer_i]  - nCovered_pool_h
                local dropped_pixels_w = nOut_conv_w[layer_i]  - nCovered_pool_w

                if (dropped_pixels_h > 0 or dropped_pixels_w > 0) then
                    if zeroPadForPooling then 
                        nOut_pool_h[layer_i] = math.ceil( (nOut_conv_h[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
                        nOut_pool_w[layer_i] = math.ceil( (nOut_conv_w[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1

                        local nCovered_pool_h_ext = (nOut_pool_h[layer_i] - 1) * poolStrides[layer_i] + poolSizes[layer_i]
                        local nCovered_pool_w_ext = (nOut_pool_w[layer_i] - 1) * poolStrides[layer_i] + poolSizes[layer_i]

                        local nPad_h = nCovered_pool_h_ext - nOut_conv_h[layer_i];
                        local nPad_w = nCovered_pool_w_ext - nOut_conv_w[layer_i];

                        local padTop,  padBottom = splitInTwo(nPad_w)
                        local padLeft, padRight  = splitInTwo(nPad_h)
                        --local padTop,  padBottom = nPad_h, 0
                        --local padLeft, padRight  = nPad_w, 0
                        local zeroPaddingModule = nn.SpatialZeroPadding(padLeft, padRight, padTop, padBottom)
                        --print(string.format('   output of layer %d : %dx%d\n', layer_i, nOut_pool_h[layer_i], nOut_pool_w[layer_i]))
                        --print(string.format('  >> Warning : Pooling in layer %d would drop %d from the height and %d from the width.',
                          --      layer_i, dropped_pixels_h, dropped_pixels_w))
                        --print(string.format('  >> So we are adding %d x %d of zero padding [L%d, R%d, T%d, B%d] before adding the pooling module',
                          --      nPad_h, nPad_w,padLeft, padRight, padTop, padBottom))

                        feat_extractor:add(zeroPaddingModule)


                        nOut_pool_h[layer_i] = math.ceil( (nOut_conv_h[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1
                        nOut_pool_w[layer_i] = math.ceil( (nOut_conv_w[layer_i] - poolSizes[layer_i])/poolStrides[layer_i]) + 1

                    else
                        error(string.format('Warning : Pooling in layer %d drops %d from the height and %d from the width', 
                                layer_i, dropped_pixels_h, dropped_pixels_w))
                    end

                end


                feat_extractor:add(poolingModule)


            else
                nOut_pool_h[layer_i] = nOut_conv_h[layer_i]
                nOut_pool_w[layer_i] = nOut_conv_w[layer_i]            
            end
            
            
            if doDropoutInConvLayers and (convLayerDropoutPs[layer_i] > 0) then  --  <== is this (after pooling) where to put it?
                if useSpatialDropout then
                    feat_extractor:add(  nn.SpatialDropout(   convLayerDropoutPs[layer_i] ) )                
                else
                    feat_extractor:add(  nn.Dropout(   convLayerDropoutPs[layer_i] ) )                
                end
            end
             
            local doSpatSubtrNorm_thisLayer = doSpatSubtrNorm and string.lower(spatSubtrNormType[layer_i]) ~= 'none'
            if doSpatSubtrNorm_thisLayer then
                local norm_kernel = getNormKernel(spatSubtrNormType[layer_i], spatSubtrNormWidth[layer_i])
                feat_extractor:add( nn.SpatialSubtractiveNormalization(nStatesConv[layer_i], norm_kernel ) )
            end
            
            local doSpatDivNorm_thisLayer   = doSpatDivNorm  and string.lower(spatDivNormType[layer_i]) ~= 'none' 
            if doSpatDivNorm_thisLayer then
                local norm_kernel = getNormKernel(spatDivNormType[layer_i], spatDivNormWidth[layer_i])
                feat_extractor:add( nn.SpatialSubtractiveNormalization(nStatesConv[layer_i], norm_kernel ) )
            end
            --]]

        end

        NOut_pool_h = nOut_pool_h
        NOut_pool_w = nOut_pool_w
        NStatesConv = nStatesConv
        
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

        nUnitsInLastLayer = nOutputs_last
        -- fully-connected layers (if any)
        if nFCLayers > 0 then
            for layer_i,nUnitsInThisLayer in ipairs(nStatesFC) do
                
                feat_extractor:add(  nn.Linear(nUnitsInLastLayer, nUnitsInThisLayer) )
                
                local nlin = getNonlinearity(nLinType) 
                feat_extractor:add(  nlin )

                nUnitsInLastLayer = nUnitsInThisLayer
                
                if doDropoutInFullyConnectedLayers and (fullyConnectedDropoutPs[layer_i] > 0) then --- moved  dropout to be AFTER each fully connected layer
                    feat_extractor:add(  nn.Dropout(   fullyConnectedDropoutPs[layer_i] ) )
                end

            end
        end

        --classifier 
        classifier:add(  nn.Linear(nUnitsInLastLayer, nOutputs) )

        if finalLayer == 'LogSoftMax' then
            classifier:add(  nn.LogSoftMax() )
        end

        if nPositions > 1 then
            local indiv_pos_classifier = nn.Linear(nUnitsInLastLayer, nOutputs * nPositions);

            classifier_indiv_pos:add( indiv_pos_classifier )  -- classifier for each individual position
            classifier_indiv_pos:add( nn.LogSoftMax() )

            classifier_indiv_pos_combined:add(  indiv_pos_classifier )  -- classifier for each letter, taking in inputs from each letter at each individual position
            classifier_indiv_pos_combined:add(  nn.Tanh() )
            classifier_indiv_pos_combined:add(  nn.Linear(nOutputs * nPositions, nOutputs) )
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
    M = model
    model:add(feat_extractor)
    model:add(classifier)

    --model = feat_extractor

    model_struct.model = model   
    --print('model_struct.model')
    --print(model_struct.model)
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
        --moveModelToGPU(model_struct)
        model_struct.model:cuda()
    end

    --model_struct.reshapeImage = reshapeImage
    return model_struct

end



getNonlinearity = function(nLinType) 
    if string.lower(nLinType) == 'tanh' then
        return nn.Tanh()
    elseif string.lower(nLinType) == 'relu' then
        return nn.ReLU()
    else
        error('Unknown nonlinearity type')
    end


end


moveModelToGPU = function(model_struct)

    --local cur_model_use = model_struct.model
    MS = model_struct

    if not (model_struct.model_onGPU == true) then  -- ie. model is on CPU

        -- check whether the model has SpatialConvolutionMap (which is not currently compatible with the GPU)

        if model_struct.parameters.useConnectionTable or areAnySubModulesOfType(model_struct.model, 'nn.SpatialConvolutionMap') then
            error('Cannot move model to the GPU : model uses the "SpatialConvolutionMap" module')
        end


        if not model_struct.model_onGPU then
            io.write('Moving Model to the GPU for the first time ... ')

            --model_struct.model_onCPU = model_struct.model

            --local model_onGPU = nn.Sequential()
            --model_onGPU:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
            --model_onGPU:add(model_struct.model_onCPU:cuda())
            --model_onGPU:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
            --model_struct.model_onGPU = model_onGPU

        else
            io.write('Moving model back to the GPU ... ')
            model_struct.model_onGPU.modules[2]:cuda() -- copy back to the GPU

        end

        io.write(' done.\n')

        model_struct.model = model_struct.model_onGPU

        model_struct.model_onGPU = true
    end


end

moveModelBackToCPU = function(model_struct)

    --local cur_model_use = model_struct.model
    if (model_struct.model_onGPU == true) then
        io.write('Moving model back to the CPU ... \n')
        model_struct.model = model_struct.model_onCPU:float()   -- this is the original model (without the CudaTensor <-> FloatTensor copying).

        model_struct.model_onGPU = false
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


getDefaultConvNetParams = function(defaultNetworkName)
    
    --assert(defaultNetworkName)
    defaultNetworkName = defaultNetworkName or 'LeNet'
        
    local params = {}
    if defaultNetworkName == 'LeNet' then
    
        params.netType = 'ConvNet'
        params.defaultNet = defaultNetworkName
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
    
    
    elseif defaultNetworkName == 'FHWA_default' then
    
        params.netType = 'ConvNet'
        params.defaultNet = defaultNetworkName
        params.nStatesConv = {128,256,512,512}
        params.nStatesFC = {1024}    
        params.convFunction = 'SpatialConvolutionMap'
        
        params.fanin = {1,4,4,4}
        params.filtSizes = {9,9,9,5}

        params.doPooling = true
        params.poolSizes = {2}
        params.poolTypes = {2}
        params.poolStrides = 'auto' --{2,2}

        params.doSpatSubtrNorm = true
        params.spatSubtrNormType = 'gauss'
        params.spatSubtrNormWidth = 7
        
        params.doSpatDivNorm = false    

    end
    
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
            w2:copy(w1)

            local b1 = full_model1.modules[net1_idx].bias
            local b2 = full_model2.modules[net2_idx].bias

            io.write(string.format('Copying bias #%d from network1 (size=%s) to network2 (size=%s) \n\n', 
                    convLayer_idx, tostring_inline(b1:size()), tostring_inline(b2:size()) ) )
            --copyConvolutionalWeights(full_model1, net1_idx, full_model2, net2_idx)
            b2:copy(b1)
            --full_model1.modules[net1_idx].weight:copy(full_model2.modules[net2_idx].weight)

        elseif net1_idx and not net2_idx then
            print('Network 1 has a Conv filter #' .. convLayer_idx .. ', but network #2 does not...')

        elseif not net1_idx and net2_idx then
            print('Network 1 does not have a Conv filter #' .. convLayer_idx .. ', but network #2 does...')

        end    

        convLayer_idx = convLayer_idx + 1

    until (not net1_idx and not net2_idx)



end



areConvolutionalWeightsTheSame = function(model_struct1, model_struct2)

    local full_model1 = getStreamlinedModel(model_struct1.model)
    local full_model2 = getStreamlinedModel(model_struct2.model)


    local net1_idx, net2_idx, convLayer_idx
    convLayer_idx = 1

    local val1,val2


    repeat 
        net1_idx = getModuleIndex(full_model1, 'Conv' .. convLayer_idx)
        net2_idx = getModuleIndex(full_model2, 'Conv' .. convLayer_idx)
        if net1_idx and net2_idx then
            local w1 = full_model1.modules[net1_idx].weight
            local w2 = full_model2.modules[net2_idx].weight
            local b1 = full_model1.modules[net1_idx].bias
            local b2 = full_model2.modules[net2_idx].bias


            if not val1 then
                val1 = w1:storage()[w1:storageOffset()]
                val2 = w2:storage()[w2:storageOffset()]
            end
            if not torch.tensorsEqual(w1, w2) or not torch.tensorsEqual(b1, b2) then    
                return false, val1, val2
            end

            convLayer_idx = convLayer_idx + 1
        end
    until (not net1_idx and not net2_idx)

    assert(val1 == val2)
    return true, val1, val2

end


firstConvolutionalWeightValue = function(model_struct1)

    full_model1 = getStreamlinedModel(model_struct1.model)    

    local net1_idx = getModuleIndex(full_model1, 'Conv1')
    if net1_idx then
        local val1 = full_model1.modules[net1_idx].weight:storage()[1]
        local val2 = full_model1.modules[net1_idx].weight:storage()[2]
        --return val1, val2        
        return val1
    end

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


replaceClassifierLayer = function(model_struct, nOutputs)

    local full_model = getStreamlinedModel(model_struct.model)

    local classifier_idx = getModuleIndex(full_model, 'classifier')

    local nModules = #(full_model.modules)

    print(string.format('Replacing the classifier with a new one with %d classes', nOutputs))

    local model_new = nn.Sequential()

    for layer_i, curModule in ipairs(full_model.modules) do

        if layer_i == classifier_idx then

            assert(torch.typename(curModule) == 'nn.Linear')
            local nOutputsPrev = curModule.weight:size(1)
            local nUnitsInLastLayer = curModule.weight:size(2)

            --local nUnitsInLastLayer = prevModule.output:size()[1]
            local newClassifierModule = nn.Linear(nUnitsInLastLayer, nOutputs)
            if curModule.bias:type() == 'torch.CudaTensor' then
                newClassifierModule:cuda() 
            end
            io.write(string.format('  [Old classifier was %d => %d. New classifier is %d => %d]\n',  
                    nUnitsInLastLayer, nOutputsPrev,   nUnitsInLastLayer, nOutputs )) 

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


splitModelFromLayer = function(model_struct, splitLayer, splitAfterFlag) 
    -- default is to split just BEFORE the specified layer
    -- add a third argument as a flag to split just AFTER that layer
    
    local full_model = getStreamlinedModel(model_struct.model)
    local addCopyUnitsForGPUmodel = false
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
    if splitAfterFlag then
        print(string.format('\nSplitting the network just after %s (layer #%d)\n', splitLayer_str, splitLayer))
        splitLayer = splitLayer + 1
    else
        print(string.format('\nSplitting the network just before  %s (layer #%d)\n', splitLayer_str, splitLayer))
    end

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




fixTopLayer = function ( topModule, nOutputs, nPositions )

    local bias = topModule.bias
    local weight = topModule.weight

    --print(bias:size(1), nOutputs)
    assert(bias:size(1) == nOutputs)
    assert(weight:size(1) == nOutputs)
    assert(weight:size(2) == nOutputs*nPositions)
    for i = 1,nOutputs do
        bias[i] = 0

        for j = 1,nOutputs * nPositions do
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

            Module_i = module_i
            local module_full_str = torch.typename(module_i)
            --print(module_full_str )
            assert(string.sub(module_full_str, 1, 3) == 'nn.')

            local module_str = string.sub(module_full_str, 4, #module_full_str) -- remove 'nn.' prefix.
            
            --[[    no longer necessary if we use torch.typename instead of tostring
            local idx_bracket = string.find(module_str, '[(]') -- remove parenthetical descriptors ( e.g. SpatialConvolution(1 -> 16, 5x5)  )
            if idx_bracket then
                module_str = string.sub(module_str, 1, idx_bracket-1)
            end
            --]]
            

            local j = net.nModules

            if string.sub(module_str, 1, 10) == 'Sequential' then
--                net.networkStr = net.networkStr .. module_full_str .. '

                net = addModules(net, module_i.modules)

            else 
                net[ 'm' .. j .. '_str'] = str2vec(module_str)
                local module_name_str = module_str

                local requiredFieldNames = {}
                local optionalFieldNames = {}
                if module_str == 'SpatialConvolution' or  module_str == 'SpatialConvolutionCUDA'  then
                    
                    requiredFieldNames = {'bias', 'weight', 'nInputPlane', 'nOutputPlane', 'kH', 'kW', 'dH', 'dW'}
                    optionalFieldNames = {'padH', 'padW'}
                    module_name_str = 'Conv'
                    
                elseif module_str == 'SpatialConvolutionMap' then
                    
                    requiredFieldNames = {'bias', 'weight', 'nInputPlane', 'nOutputPlane', 'kH', 'kW', 'dH', 'dW', 'connTable'}  -- connTable optional?
                    module_name_str = 'Conv'
                    
                elseif module_str == 'SpatialSubSampling' then        
                
                    requiredFieldNames = {'kH', 'kW', 'dH', 'dW', 'connTable'}
                    module_name_str = 'SubSamp'

                elseif module_str == 'SpatialAveragePooling' then   
                
                    requiredFieldNames = {'kH', 'kW', 'dH', 'dW', 'divide'}    
                    module_name_str = 'SubSamp'                
                
                elseif module_str == 'MulConstant' then   
                
                    requiredFieldNames = {'inplace', 'constant_scalar'}    
                    module_name_str = 'MulConst'     
                
                elseif module_str == 'SpatialZeroPadding' then                
                    
                    requiredFieldNames = {'pad_l', 'pad_r', 'pad_t', 'pad_b'}
                    module_name_str = 'ZeroPad'

                --elseif (module_str == 'SpatialMaxPooling') or (module_str == 'SpatialMaxPoolingCUDA') then                    
                elseif module_str == 'SpatialMaxPooling' then
                
                    requiredFieldNames = {'kH', 'kW', 'dH', 'dW', 'indices'}   -- indices optional?
                    optionalFieldNames = {'padH', 'padW', 'ceil_mode'}
                    module_name_str = 'MaxPool'
                
                elseif module_str == 'Reshape' then
                
                    requiredFieldNames = {'nelement', 'vector', 'size', 'batchsize'}
                    module_name_str = 'Reshape'
                
                elseif module_str == 'Linear' then

                    requiredFieldNames = {'bias', 'weight'}
                    optionalFieldNames = {'addBuffer'}
                    module_name_str = 'Linear'
                
                elseif module_str == 'Transpose' then
                    requiredFieldNames = {'permutations'}

                elseif (module_str == 'Sqrt') then
                    requiredFieldNames = {'eps'}

                elseif (module_str == 'Dropout') or (module_str == 'SpatialDropout')  then
                    requiredFieldNames = {'p', 'noise'}
                    optionalFieldNames = {'v2'}  -- for regular dropout
                
                elseif (module_str == 'Square') or (module_str == 'Sqrt') or (module_str == 'Copy') or 
                    (module_str == 'Tanh') or (module_str == 'LogSoftMax') or (module_str == 'Exp')  then     
                    
                else                
                    error('Unhandled case : module type = ' .. module_str)

                end
                
                optionalFieldNames = table.merge(optionalFieldNames, {'train'})
            

            
                -- check that didn't leave any fields out:
                local allfieldnames_have = table.fieldnames(module_i)
                H = allfieldnames_have
                local fieldNames_copied = table.merge(requiredFieldNames, optionalFieldNames)
                allfieldnames_have = table.setdiff(allfieldnames_have, {'_input', 'gradInput', 'finput', 'fgradInput', 
                        'output', '_gradOutput', 'gradBias','gradWeight' })
                
                local extraFields_lookedFor = table.setdiff(requiredFieldNames, allfieldnames_have)
                local missingFields_notCopied = table.setdiff(allfieldnames_have, table.merge(requiredFieldNames, optionalFieldNames))
                
                if #extraFields_lookedFor > 0 then
                    print('Module ', mod_j, module_full_str, 'These fields were looked for, but not not present : ', extraFields)
                    error('Some fields were not present')
                end
                
                if #missingFields_notCopied > 0 then
                    print('Module ', mod_j, module_full_str, 'These fields were left out : ', missingFields_notCopied)
                    error('Some fields were left out')
                end
                
                net.modules_str = net.modules_str .. module_name_str .. ';' 
                net.nModules = net.nModules + 1


            
                for j = 1,2 do
                    local fldNames, required
                    if j == 1 then
                        fldNames = requiredFieldNames
                        required = true
                    elseif j == 2 then
                        fldNames = optionalFieldNames
                        required = false
                    end

                    for i,fieldName in ipairs(fldNames) do
                        local fieldVal = module_i[fieldName]
                        local globFieldName = 'm' .. j .. '_' .. fieldName
                        
                        if fieldVal == nil then
                            if required then 
                                error(string.format('Module %s does not have field %s', module_str, fieldName))
                            end
                        elseif torch.isTensor(fieldVal) then
                            net[globFieldName] = module_i[fieldName]:double()
                        elseif type(fieldVal) == 'number' then
                            net[globFieldName] = torch.DoubleTensor(fieldVal)
                        elseif torch.isStorage(fieldVal) then
                            local double_storage = torch.DoubleStorage(fieldVal:size()):copy(fieldVal)
                            net[globFieldName] = torch.DoubleTensor(double_storage)
                        elseif type(fieldVal) == 'boolean' then
                            local num = iff(fieldVal, 1, 0)
                            net[globFieldName] = torch.DoubleTensor(num)
                        else
                            error(string.format('Unhandled type : %s is a %s [%s]', fieldName, type(fieldVal), tostring(fieldVal)))
                        end
                    
                    end
                end
                
                
              
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





