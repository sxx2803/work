
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity CompTB is
end entity CompTB;

architecture CompTB of CompTB is
	
component comparator
	
generic(nInputs   : integer := 9);
port (
		clk     :	in std_logic;
		input     :	in std_logic_vector(nInputs downto 1);
		output  :	out std_logic
	);
end component comparator;

signal		sclk     :	 std_logic := '0';
signal		sinput   :	 std_logic_vector(9 downto 1);
signal		soutput  :	 std_logic;

begin
uut1: comparator port map(clk=>sclk, input=>sinput, output=>soutput);

sclk <= not sclk after 10 ns;

process
	begin
	wait for 20 ns;
	sinput <= "100110000";
	wait for 20 ns;
	report std_logic'image(soutput);
	wait for 20 ns;
	sinput <= "000110000";
	wait for 20 ns;
	sinput <= "100010000";
	wait for 20 ns;
	sinput <= "000000000";
	wait for 20 ns;
	wait;
end process;	
--  
--component comparator
--	generic(nInputs         : integer := 9);
--  	port(
--  		clk : in std_logic; 
--  		input : in std_logic_vector(nInputs downto 1); 
--        output    : out std_logic); 
--end component comparator;
--
--signal sinput: std_logic_vector(9 downto 1);
--signal soutput: std_logic;
--signal sclk: std_logic;
--
--begin
--	uut1: comparator port map(clk=>sclk, input=>sinput, output=>soutput);
--	sclk <= not sclk after 10 ns;
--	sinput <= "100110000";
--	
end architecture CompTB;
