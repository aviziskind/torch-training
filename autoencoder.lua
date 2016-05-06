applyAutoencoderToDataset = function(model, images, overlap_frac)
    
    nImages = images:size(1)
    output_images = torch.Tensor(images:size()):typeAs(images)
    for i = 1,nImages do
        xx = applyAutoencoderToImage(model, images[{{i,i}}], overlap_frac)
        output_images[{{i,i}}] = xx
    end
        
    return output_images
    
end


lastElements = function(t, n)
    local nTot 
    if torch.isTensor(t) then
        nTot = t:numel()
    else
        nTot = #t
    end
    
    n = n or 1
    n = math.min(n, nTot)
    
    
    if nTot >= 2 then
        if torch.isTensor(t) then
            return t[{{nTot-n+1, nTot}}]
        elseif torch.isStorage(t) then
            
            return torch.LongStorage(t, nTot-n+1, n)
        end
    end
    return t
end

applyAutoencoderToImage = function(model, image, overlap_frac)
    M = model
    I = image
    modelInputSize = lastElements( model.encoder.modules[1].gradInput:size(), 2)
    imageSize = lastElements( image:size(), 2)
    
    if modelInputSize[1] == imageSize[1] and  modelInputSize[2] == imageSize[2] then
        model:forward(image)
        return model.decoder.D.output
    end
    
    local count = torch.Tensor(image:size()):typeAs(image):fill(1e-5)
    local output = torch.Tensor(image:size()):typeAs(image):fill(0)
    Count = count   
    
        
    Output = output
    
    
    overlap_frac = overlap_frac or 0.5
    
    scl = math.min( math.max( (1-overlap_frac), 1e-5), 1-1e-5)
    
    i_step = math.ceil(modelInputSize[1] * scl )
    j_step = math.ceil(modelInputSize[2] * scl )
    i_max = imageSize[1]-modelInputSize[1]+1
    j_max = imageSize[2]-modelInputSize[2]+1
    idx_starts_i = torch.range(1, i_max, i_step)
    idx_starts_j = torch.range(1, j_max, j_step)

    if idx_starts_i[-1] < i_max then
        idx_starts_i = torch.concat( idx_starts_i, i_max)        
    end
    if idx_starts_j[-1] < j_max then
        idx_starts_j = torch.concat( idx_starts_j, j_max)        
    end

    nTot = idx_starts_i:numel()*idx_starts_j:numel()
    print('nTot', nTot, 'step', i_step)
    progressBar.init(nTot, nTot)
    for i = 1,idx_starts_i:numel() do
        idx_i = {idx_starts_i[i], idx_starts_i[i]+modelInputSize[1]-1}
        for j = 1,idx_starts_j:numel() do
            idx_j = {idx_starts_j[j], idx_starts_j[j]+modelInputSize[1]-1}

            local img_idx_tbl = {{1}, idx_i,idx_j}
            Im = image[img_idx_tbl]
            model:forward(Im )
           
            output[img_idx_tbl] = output[img_idx_tbl] + model.decoder.D.output
           
            count[img_idx_tbl] = count[img_idx_tbl] + 1
            progressBar.step()
        end
    end
        
    output:cdiv(count)
        
    return output
    
    
end


