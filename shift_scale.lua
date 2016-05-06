function modifyImageAndOutputs(im, pos, ops)
    print('start pos_rel : \n', pos)
    print('start pos_pix : \n', rel2pix(pos, im:size()), 'image size : \n', im:size())

    if #ops == 0 then
        ops = {ops}
    end

    local nOps = #ops

    for i, op in ipairs(ops) do
        if table.anyEqualTo(op.type, {'crop', 'pad', 'cropOrPad'}) then
            
            im, pos = cropOrPadImageAndPosition(im, pos, op)
            print('just got rel pos = ', pos)
        elseif (op.type == 'resize') then
            im = resizeImage(im, op);  -- pos stays the same!
        else
            error('Unknown image modification type : ' .. op.type);
        end
                
    end
    
    print('new pos_rel : \n', pos)
    print('new pos_pix : \n', rel2pix(pos, im:size()), 'image size : \n', im:size())
    return im, pos
    
end




function cropOrPadImageAndPosition(cur_image, pos_rel, opt)
--         op{1} = struct('type', 'crop',   'edge', 'random',  'frac', 'random', 'fracMax', 0.3, 'prob', .2);
--         op{3} = struct('type', 'cropOrPad', 'size', size(X), 'padStyle', 'random', 'padValue', 0);

        
-- --     newSize, cropStyle, padStyle, padValue, jointBorder
--     newSize, padStyle, padValue
--     
    local curSize = cur_image:size()
    pos_rel = pos_rel:reshape(pos_rel:numel()/2, 2)

    local new_image = cur_image
    local new_pos_rel = pos_rel
    
    if opt.prob and opt.prob < 1 then
       if torch.uniform() > opt.prob then   -- only do the cropping/padding with this probability. Otherwise, skip
           return new_image, new_pos_rel
       end
    end

    
    local  padStyle_h, padStyle_w = 'random', 'random'
    if opt.padStyle and opt.padStyle ~= '' then
        padStyle_h, padStyle_w = opt.padStyle, opt.padStyle
    end
    if opt.padStyle_h and opt.padStyle_h ~= '' then
        padStyle_h = opt.padStyle_h;
    end
    if opt.padStyle_w and opt.padStyle_w ~= '' then
        padStyle_w = opt.padStyle_w;
    end
        
    
    local  cropStyle_h, cropStyle_w = 'random', 'random'
    if opt.cropStyle and opt.cropStyle ~= '' then
        cropStyle_h, cropStyle_w = opt.cropStyle, opt.cropStyle
    end
    if opt.cropStyle_h and opt.cropStyle_h ~= '' then
        cropStyle_h = opt.cropStyle_h;
    end
    if opt.cropStyle_w and opt.cropStyle_w ~= '' then
        cropStyle_w = opt.cropStyle_w;
    end
        
    
    local padValue = 0
    if opt.padValue then 
        padValue  = opt.padValue
    end

    local jointBorder = 0.05;
    if opt.jointBorder then
        jointBorder = opt.jointBorder;
    end
    
    local newSize
    if opt.size then
        newSize = opt.size;
        
    elseif opt.edge then
        
        local allEdges = 'LRTB' 
        
        local edgeUse = opt.edge;
        local idx_edgeToUse
        if edgeUse == 'random' then
            idx_edgeToUse = torch.random(4)
        elseif type(edgeUse) == 'string' then
            if #edgeUse > 1 then -- eg 'LR' or 'LRT'
                local j = torch.random(#edgeUse)
                edgeUse = string.sub(edgeUse, j, j)  
            end
            idx_edgeToUse = string.find(allEdges, edgeUse)
            assert(idx_edgeToUse)
        elseif type(edgeUse) == 'number' then
            idx_edgeToUse = edgeUse;
        end
            
        local edgeFrac = opt.frac
        if edgeFrac == 'random' then
            edgeFrac = torch.uniform()*opt.fracMax;
        end
        
        
        newSize = torch.LongStorage(#curSize):copy(curSize)
        local idx_dim
        if idx_edgeToUse <= 2 then
            idx_dim = 2;  -- MATLAB
            --idx_dim = 1  -- torch
        elseif idx_edgeToUse >= 3 then
            idx_dim = 1;  -- MATLAB
            --idx_dim = 2  -- torch
        end

        local npix = math.round(edgeFrac * cur_image:size(idx_dim))
        local sgn
        --print('edgeFrac', edgeFrac, 'npix', npix)
        
        local padStyle = torch.zeros(4)
        local cropStyle = torch.zeros(4)
        if opt.type == 'crop' then
            cropStyle[idx_edgeToUse] = npix;
            sgn = -1
        elseif opt.type == 'pad' then
            padStyle[idx_edgeToUse] = npix;
            sgn = 1;
        else
            error('Specify whether to crop or pad the edge')
        end
        padStyle_w = padStyle[{{1,2}}]    -- MATLAB: 1,2. torch:   3,4
        padStyle_h = padStyle[{{3,4}}]    -- MATLAB: 3,4. torch:   1,2
        cropStyle_w = cropStyle[{{1,2}}]  -- MATLAB: 1,2. torch:   3,4
        cropStyle_h = cropStyle[{{3,4}}]  -- MATLAB: 3,4. torch:   1,2

        newSize[idx_dim] = newSize[idx_dim] + sgn * npix;
        --print('newSize', newSize, 'curSize', curSize)
        
    else 
        error('Specify a new size, or an edge to crop/pad')
    end
    
    
    local dH = newSize[1] - curSize[1];  -- MATLAB : h-->1  w-->2
    local dW = newSize[2] - curSize[2];  -- torch  : h-->2  w-->1
    --cprintf.cyan('dH, dW = %d, %d\n', dH, dW)

    local pad_l,  pad_r,  pad_t,  pad_b  = 0,0,0,0
    local crop_l, crop_r, crop_t, crop_b = 0,0,0,0
    
    if dH > 0 then  -- pad
        pad_t, pad_b   = splitAccordingToStyle(dH, padStyle_h)
    elseif dH < 0 then -- crop
        crop_t, crop_b = splitAccordingToStyle(-dH, cropStyle_h)
    end
        
    --print('cropStyle_w', cropStyle_w, 'cropStyle_h', cropStyle_h)
    if dW > 0 then -- pad
        pad_l, pad_r = splitAccordingToStyle(dW, padStyle_w)
    elseif dW < 0 then  -- crop
        crop_l, crop_r = splitAccordingToStyle(-dW, cropStyle_w)
    end
    
    cprintf.cyan('pad [l,r,t,b] = %d, %d, %d, %d\n', pad_l, pad_r, pad_t, pad_b)
    cprintf.cyan('crop [l,r,t,b] = %d, %d, %d, %d\n', crop_l, crop_r, crop_t, crop_b)
       
    if pad_l>0 or pad_r>0  or pad_t>0  or pad_b>0  then
        new_image = padarray2(new_image, pad_l, pad_r, pad_t, pad_b, padValue)
    end
    
    if crop_l>0 or crop_r>0  or crop_t>0  or crop_b>0  then
        new_image = croparray(new_image, crop_l, crop_r, crop_t, crop_b)
    end
    
    
    local pos_pix_unpadded = rel2pix(pos_rel, curSize)
        
    -- adjust position based on cropping / padding    
    local pos_pix_padded   = pos_pix_unpadded;
    pos_pix_padded[{{}, 1}] = pos_pix_padded[{{}, 1}] + pad_t - crop_t  -- MATLAB: + pad_l - crop_l
    pos_pix_padded[{{}, 2}] = pos_pix_padded[{{}, 2}] + pad_l - crop_l  -- MATAB:  + pad_t - crop_t

    new_pos_rel = pix2rel(pos_pix_padded, newSize)
    

    P = pos_rel
    NP = new_pos_rel
    NS = newSize
    -- remove joints that were missing in original position, or that have
    -- now been cut off.
    idx_joints_missing =                               pos_rel:eq(-1):sum(2):view(-1):nonzero()
    idx_joints_cut_off = torch.abs(new_pos_rel):gt(0.5 + jointBorder):sum(2):view(-1):nonzero()

    N = new_pos_rel
    if idx_joints_missing:numel() > 0 then
        cprintf.red('removing missing joints\n');
        new_pos_rel:indexFill(1, idx_joints_missing:view(-1), -1)
    end
    if idx_joints_cut_off:numel() > 0 then
        cprintf.red('removing cut off joints');
        new_pos_rel:indexFill(1, idx_joints_cut_off:view(-1), -1)
    end
                
    return new_image, new_pos_rel
        
    
end

function splitAccordingToStyle(dX, splitStyle)
    local a,b
    assert(splitStyle)
    
    if type(splitStyle) == 'table' or torch.isTensor(splitStyle) then
        a = splitStyle[1]
        b = splitStyle[2];
        --assert(#splitStyle == 2)
        assert(a+b == dX)
        
    elseif type(splitStyle) == 'string' then
        if splitStyle == 'random' then
            a = torch.random(dX+1)-1; b = dX-a;
        elseif splitStyle == 'max' then 
            a = dX; b = 0;
        elseif splitStyle == 'min' then  
            a = 0; b = dX;
        elseif splitStyle == 'even' then 
            a,b = math.splitInTwo(dX);
        else
            error('Unknown style of splitting')
        end
    end
    
    return a,b
end
        
    



function resizeImage(cur_image, opt)

    assert(opt.type == 'resize')
    
    local new_image = cur_image;
    
    -- determine if we are going to scale
    if opt.prob and opt.prob < 1 then
       if torch.uniform() > opt.prob then   -- only do the cropping/padding with this probability. Otherwise, skip
           return cur_image
       end
    end

    -- get scale factor
    local scale_factor
    if (opt.scale_factor == 'random') then
        scale_factor = opt.scale_factor_min + rand*(opt.scale_factor_max - opt.scale_factor_min);
        
    elseif type(opt.scale_factor) == 'table' then        
        scale_factor = opt.scale_factor[1] + torch.uniform()*(opt.scale_factor[2]-opt.scale_factor[1])
        
        assert(#opt.scale_factor == 2)        
    elseif type(opt.scale_factor) == 'number' then
        scale_factor = opt.scale_factor;
        
    end
    
    if scale_factor ~= 1 then
        new_image = image.scale(cur_image, '*' .. scale_factor);
    end
  
    return new_image

end
        
        

function croparray(cur_image, crop_l, crop_r, crop_t, crop_b)
    
    local nDims = cur_image:nDimension()
    local curSize = cur_image:size()
    if not crop_t and type(crop_l) == 'table' or torch.isTensor(crop_l) then
        crop_l, crop_r, crop_t, crop_b = crop_l[1], crop_l[2], crop_l[3], crop_l[4]
    end

    idx_i = {crop_t+1 , curSize[nDims-1]-crop_b}
    idx_j = {crop_l+1 , curSize[nDims  ]-crop_r}
    idx_tbl = table.rep({}, nDims)
    idx_tbl[nDims-1] = idx_i
    idx_tbl[nDims] = idx_j
    
    return cur_image[idx_tbl]

end





