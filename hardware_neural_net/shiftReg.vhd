library ieee; 
use ieee.std_logic_1164.all; 
use ieee.numeric_std.all;
 
entity weightShift is  
	
	generic(nInputs   : integer);
  	port(clk, WSI, SE : in  std_logic; 
          output      : out std_logic_vector(nInputs downto 1)); 
end weightShift; 

architecture behavioral  of weightShift is 
  signal reg: std_logic_vector(nInputs downto 1); 
  begin 
    process (clk) 
      begin  
        if (clk'event and clk='1' and SE='1') then
            reg <= to_StdLogicVector(to_bitvector(reg) sll 1);
            reg(1) <= WSI;
        end if; 
    end process; 
    output <= reg; 
end architecture behavioral;