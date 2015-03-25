
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.numeric_std.all;
 
entity comparator is
	generic(nInputs         : integer);
  	port(
  		clk : in std_logic; 
  		input : in std_logic_vector(nInputs downto 1); 
        output    : out std_logic); 
end comparator; 

architecture behavioral  of comparator is 
  
signal thresh: std_logic_vector(nInputs downto 1) := "000000110011"; 
 
begin 
    process (clk) 
      begin  
      	if (clk'event and clk = '1') then
      		case signed(input) >= signed(thresh) is
		        when true => output <= '1';
		        when false => output <= '0';
		    end case;
        end if; 
    end process;
end architecture behavioral; 