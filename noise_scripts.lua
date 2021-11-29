getFilterStr = function(filt, wForWhite)
    
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
        
        if filt.cycPerLet_centFreq then                
            filtStr = string.format('Nband%.0f',     filt.cycPerLet_centFreq*10)
        elseif filt.pixPerCycle_centFreq then
            filtStr = string.format('Nband%.0fppc', filt.pixPerCycle_centFreq)               
        elseif filt.cycPerPix_centFreq then                
            filtStr = string.format('Nband%.0fcpp', filt.cycPerPix_centFreq)
        else
            error('No center frequency field')
        end
        
    elseif filt.filterType == 'hi' then
        filtStr = string.format('Nhi%.0f', filt.cycPerLet_cutOffFreq*10)
    
    elseif filt.filterType == 'lo' then
        filtStr = string.format('Nlo%.0f', filt.cycPerLet_cutOffFreq*10)


    elseif string.sub(filt.filterType, 1, 3) == '1/f' then
   
        local f_exp_std_str = ''
        if filt.f_exp_std and filt.f_exp_std > 0 then
            f_exp_std_str = string.format('s%.0f', filt.f_exp_std*100)
        end
    

        if filt.filterType == '1/f' then
            filtStr = string.format('Npink%.0f%s', filt.f_exp*10, f_exp_std_str)
        
        elseif (filt.filterType == '1/fPwhite') or  (filt.filterType == '1/fOwhite' ) then
        
            local f_exp = filt.f_exp;
            local f_exp_default = 1.0;
            local f_exp_str = '';
            local pinkWhiteRatio = filt.ratio;
                
            local pinkExtraStr = '';
            local whiteExtraStr = '';
            if pinkWhiteRatio > 1 then
                pinkExtraStr = string.format('%.0f', pinkWhiteRatio * 10);
            elseif pinkWhiteRatio < 1 then
                whiteExtraStr = string.format('%.0f', (1/pinkWhiteRatio)*10 );
            end
            
            if f_exp ~= f_exp_default then
                f_exp_str = string.format('%.0f', f_exp*10);
            end
            
            local plus_or_str = switchh(filt.filterType, {'1/fPwhite','1/fOwhite'}, {'P', 'O'});
           
            filtStr = string.format('N%spink%s%s%sw%s', pinkExtraStr, f_exp_str, f_exp_std_str, 
                plus_or_str, whiteExtraStr);    
        end
    
    else
        error(string.format('Unknown filter type: %s ', filt.filterType))
    end
    
    
    local radius_str = ''
    if filt.radius then
        radius_str = string.format('_rad%d', filt.radius)    
    end
    
    return filtStr .. normStr .. radius_str
    

end

