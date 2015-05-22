require 'optim'

if (Logger2 == nil) then
    Logger2 = torch.class('optim.Logger2')
end

function Logger2:__init(filename, open_mode)
    if filename then
        self.name = filename
        os.execute('mkdir -p "' .. sys.dirname(filename) .. '"')
        --if timestamp then
            -- append timestamp to create unique log file
            --filename = filename .. '-'..os.date("%Y_%m_%d_%X")
        --end
        if not open_mode or open_mode == 'new' then
            self.file = io.open(filename,'w')
            self.empty = true
        else
            self.file = io.open(filename,'a')
            self.empty = false
        end
    else
        self.file = io.stdout
        self.name = 'stdout'
        print('<Logger> warning: no path provided, logging to std out') 
    end
    
    self.symbols = {}
    self.nsymbols = 0
    self.fieldOrder = {}
end


function Logger2:add(symbols)
    self.fieldOrder = self.fieldOrder or {}
    for k,val in pairs(symbols) do
        newField = self.symbols[k] == nil
        self.symbols[k] = val
        if newField then
            self.fieldOrder[#self.fieldOrder + 1] = k
        end
    end
    
end


function Logger2:println()
--      print(self.symbols)
  -- (1) first time ? print symbols' names on first row
    self.nsymbols = #(self.fieldOrder)
    if self.empty then
        self.empty = false
       
        for i,fld_name in ipairs(self.fieldOrder) do
            if string.find(fld_name, ' ') then
                fld_name = '"' .. fld_name .. '"'
            end
                
            self.file:write(fld_name)
            if (i<self.nsymbols) then
                self.file:write(',')
            end
        end
        self.file:write('\n')
    end
    
    -- (2) print all symbols on one row
    for i,fld_name in ipairs(self.fieldOrder) do
    
        self.file:write(self.symbols[fld_name])
        if (i<self.nsymbols) then
            self.file:write(',')
        end
        
    end
    self.file:write('\n')
    self.file:flush()
end

-----------------

function Logger2:style(symbols)
    for name,style in pairs(symbols) do
        if type(style) == 'string' then
            self.styles[name] = {style}
        elseif type(style) == 'table' then
            self.styles[name] = style
        else
            xlua.error('style should be a string or a table of strings','Logger')
        end
    end
end


function Logger2:plot(...)
    if not xlua.require('gnuplot') then
        if not self.warned then 
            print('<Logger> warning: cannot plot with this version of Torch') 
            self.warned = true
        end
        return
    end
    local plotit = false
    local plots = {}
    local plotsymbol = function(name,list)
        if #list > 1 then
            local nelts = #list
            local plot_y = torch.Tensor(nelts)
            for i = 1,nelts do
            plot_y[i] = list[i]
            end
            for _,style in ipairs(self.styles[name]) do
            table.insert(plots, {name, plot_y, style})
            end
            plotit = true
        end
    end
    
    local args = {...}
    if not args[1] then -- plot all symbols
        for name,list in pairs(self.symbols) do
            plotsymbol(name,list)
        end
    else -- plot given symbols
        for i,name in ipairs(args) do
            plotsymbol(name,self.symbols[name])
        end
    end
    if plotit then
        self.figure = gnuplot.figure(self.figure)
        gnuplot.plot(plots)
        gnuplot.title('<Logger::' .. self.name .. '>')
        if self.epsfile then
            os.execute('rm -f "' .. self.epsfile .. '"')
            local epsfig = gnuplot.epsfigure(self.epsfile)
            gnuplot.plot(plots)
            gnuplot.title('<Logger::' .. self.name .. '>')
            gnuplot.plotflush()
            gnuplot.close(epsfig)
        end
    end
end


