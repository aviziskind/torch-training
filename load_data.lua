



require 'mattorch'

convertMatToTorch = function(matfile, torchfile, convertOpts)
    
    local matfile_base = paths.basename(matfile)
    local torchfile_base = paths.basename(torchfile)
    
    print(string.format('   Converting :\n      %s ==>\n      %s...', matfile_base, torchfile_base))
    
    local S_mat = mattorch.load(matfile)    
   
    for k,v in pairs(S_mat) do 
        if (getType(v) == 'torch.CharTensor') then
            local v_str = char2string(v)
            S_mat[k] = v_str
        end            
    end
    
    if S_mat.inputMatrix then
        S_mat.inputMatrix = S_mat.inputMatrix:float()  -- SVHN/CIFAR inputMatrix is in uint8 (byte) format
    end
    
    --[[
    if convertOpts.doOverFeat then
        S_mat.inputMatrix = convertToOverFeatFeatures(S_mat.inputMatrix, convertOpts)
    end
--]]
    
    if convertOpts and convertOpts.globalNorm and not convertOpts.doTextureStatistics then   -- (do this for SVHN)
        globalNormalizeDataset(S_mat)
    end
    if convertOpts and convertOpts.elementNorm and not convertOpts.doTextureStatistics then   
        elementWiseNormalizeDataset(S_mat)
    end
    if convertOpts and convertOpts.localContrastNorm and not convertOpts.doTextureStatistics then
        localContrastNormalizeDataset(S_mat)
    end
    
    local torch_dirname = paths.dirname(torchfile)
    paths.createFolder(torch_dirname)
   
    torch.save(torchfile, S_mat)

end 

globalNormalizeDataset = function(S_data, meanVal, stdVal)
    SS_data = S_data
    io.write(' -- Normalizing inputs (subtracting global mean , dividing by global std. deviation)')

    local X = S_data.inputMatrix            
    --X:add( - X:min() )   -- subtract minimum
    --X:div( X:max() )     -- divide by max
    
    meanVal = meanVal or X:mean()  -- if not provided, calculate now
    stdVal  = stdVal  or X:std()

    X:add( - meanVal )   -- subtract mean
    X:div( stdVal )      -- divide by std
    io.write(' (done)\n')
end
    
elementWiseNormalizeDataset = function(S_data)
    -- normalize each element separately (do this for texture statistics)
    io.write(' -- Normalizing each element of the input separately...')
    local stat_mean = {}
    local stat_std = {}
    local inputMatrix = S_data.inputMatrix
    
    local width_here = inputMatrix:size(3)
    assert(width_here > 1);  assert(inputMatrix:size(2) == 1)    -- ie. column vector (for texture statistics)
    for hi = 1, width_here do
       stat_mean[hi] = inputMatrix[{ {},{},{hi} }]:mean()
       stat_std[hi] = math.max( inputMatrix[{ {},{},{hi} }]:std(), 1e-5)
       inputMatrix[{ {},{},{hi} }]:add(-stat_mean[hi])
       inputMatrix[{ {},{},{hi} }]:div(stat_std[hi])
    end
    io.write(' (done)\n')
        --]]
            
end

localContrastNormalizeDataset = function(S_data)
    -- Local contrast normalization
    io.write(' -- Local contrast normalization on each input image...')

    -- Define the normalization neighborhood:
    local neighborhood = image.gaussian1D(7)

    -- Define our local normalization operator (It is an actual nn module, 
    -- which could be inserted into a trainable model):
    local normalization = nn.SpatialContrastiveNormalization(1, neighborhood):float()

    -- Normalize all Y channels locally:
    local inputMatrix = S_data.inputMatrix
    local nSamples = inputMatrix:size(1)
    local nDim = #inputMatrix:size()
    assert(nDim == 3 or nDim == 4)
    if nDim == 4 then assert(inputMatrix:size(2) == 1) end
    
    progressBar.init(nSamples, 30)
    for i = 1,nSamples do

        if nDim == 3 then        
            inputMatrix[{   {i},{},{} }] = normalization( inputMatrix[{   {i},{},{} }] )
        elseif nDim == 4 then
            inputMatrix[{ i,{1},{},{} }] = normalization( inputMatrix[{ i,{1},{},{} }] )
        end
        progressBar.step(i)

    end
            
    io.write(' (done)\n')
   
end



function loadFile(filename, deleteIfCantLoad)
    local S = nil
    local maxNTries = 3
    
    local skip = false
    ---[[
    --cprintf.Red('%s\n', filename)
    --]]
    
    local isTorchFile = string.find(filename, '.t7') ~= nil
    local isMatFile = string.find(filename, '.mat') ~= nil
    assert( isMatFile ~= isTorchFile )
    
    
    local nTries = 0
    local function loadFileToS()
        if isTorchFile then
            S = torch.load( filename )
        elseif isMatFile then
            S = mattorch.load( filename )
            if S.inputMatrix then
                S.inputMatrix = S.inputMatrix:float()
            end
            if S.labels then
                S.labels = S.labels:float()
            end
        end
    end
    local status, result = false, nil
    while (nTries < maxNTries) and (status == false) do
        status, result = pcall( loadFileToS )
            
        if not status then -- failed
            io.write(string.format('Status = "%s"\n', result))
            if (result == 'stop') or string.find(result, 'interrupted!') then
                print('Received stop signal from user. Aborting....');
                error('Aborted by user')
            end
            local sec_wait = 3 + (torch.random() % 10)
            print(string.format('Loading %s failed, trying again in %s seconds', filename, sec_wait))
            sys.sleep(sec_wait)
            nTries = nTries + 1
        end
    end
    
    if status == false and deleteIfCantLoad then
        io.write(string.format('Tried to load this file:\n   %s\nbut got this error:\n   %s\nDeleting file.\n', filename, result))    
        os.execute(string.format('rm %s', filename))
    end
        

    return S, result
end





convertAllFieldsToDouble = function(S)
    S = table.copy(S)
    for k,v in pairs(S) do
        if type(v) == 'number' then
            S[k] = torch.DoubleTensor{v}
        
        elseif (torch.typename(v) == 'torch.CharTensor') then
            S[k] = v:double()
            --S[k] = nil
        elseif (torch.typename(v) == 'torch.FloatTensor') or (torch.typename(v) == 'torch.ByteTensor') then
            S[k] = v:double()
        elseif (torch.typename(v) == 'torch.DoubleTensor')  then
            
        else
            error( string.format('Unknown type : %s', torch.typename(v) ) )
        end
    end
    return S
    
end

addInputPlaneDimension = function(X)
   if X:nDimension() == 4 then
        return X
   end
   if X:nDimension() ~= 3 then
        error('Must have 3 or 4 dimensions')
    end
   
   local nSamples = X:size(1)
   local height = X:size(2)
   local width = X:size(3)
   return X:reshape(nSamples, 1, height, width)    
end

removeInputPlaneDimension = function(X)
   if X:nDimension() == 3 then
        return X
   end
   if X:nDimension() ~= 4 then
        error('Must have 3 or 4 dimensions')
   end
   
   local nSamples = X:size(1)
   local nInputPlanes = X:size(2)
   assert(nInputPlanes == 1)
   local height = X:size(3)
   local width = X:size(4)
   return X:reshape(nSamples, height, width)
    
end


getDataStats = function(data)
    
    local stats = {};
    local X = data.inputMatrix
    XX = X
    if X:nDimension() == 4 then 
        stats.nSamples     = X:size(1)
        stats.nInputPlanes = X:size(2)
        
        stats.width  = X:size(3)
        stats.height = X:size(4)

    elseif X:nDimension() == 3 then
        stats.nSamples     = X:size(1)
        stats.nInputPlanes = 1
        
        stats.width  = X:size(2)
        stats.height = X:size(3)
    else
        error('Input dataset must have at least 3 dimensions')
    end
    
    if data.nClasses then
        stats.nClasses = data.nClasses
        if torch.isTensor(stats.nClasses) then
            stats.nClasses = stats.nClasses[1][1]
        end
    end
    if data.nOutputs then
        stats.nOutputs = data.nOutputs
        if torch.isTensor(stats.nOutputs) then
            stats.nOutputs = stats.nOutputs[1][1]
        end
    end
    
    stats.nInputs = stats.width * stats.height * stats.nInputPlanes
    St = stats
    local Y = data.outputMatrix
    YY = Y
    
    if not Y then
        --stats.nOutputs = 0
    else
        if Y:nDimension() == 2 then 
            stats.nOutputs = Y:size(2)
            assert(Y:size(1) == stats.nSamples)
        elseif Y:nDimension() == 1 then 
            stats.nOutputs = 1
            assert(Y:numel() == stats.nSamples)
        end
    end
    -- add to data
    for k,v in pairs(stats) do
        data[k] = v
    end
    return stats
end