-------------------------------------------------------------------------------
-- File         : tb_feature.vhd
-- Entity       : tb_feature
-- Architecture : tb
-- Author       : Qutaiba Saleh
-- Author       : James Mnatzaganian
-- Created      : 4/29/14
-- Modified     : 12/09/14
-- VHDL'93
-- Testbench for testing the features for the Graduate lab
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use std.textio.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;

-------------------------------------------------------------------------------
-- Testbench entity
-------------------------------------------------------------------------------

entity tb_feature is

  -----------------------------------------------------------------------------
  -- User Defined Constants
  -----------------------------------------------------------------------------
  
  -- MLP parameters
  constant Nu          : integer := 9;
  constant Nh          : integer := 20;
  constant Ny          : integer := 1;
  constant n           : integer := 7;
  constant m           : integer := 4;

  -- Simulation parameters
  constant clk_period  : time    := 10 ns;

  -- File parameters
  constant weight_path : string  := "C:\Users\sxx2803\Documents\MATLAB\QW47.dat";
  constant in_path     : string  := "C:\Users\sxx2803\Documents\MATLAB\Qfeatures47.txt";
  constant out_path    : string  := "C:\Users\sxx2803\Documents\MATLAB\fudges47.txt";
  constant img_height  : integer := 38;
  constant img_width   : integer := 9;
end tb_feature;

-------------------------------------------------------------------------------
-- Architecture defintion
-------------------------------------------------------------------------------

architecture tb of tb_feature is

  -----------------------------------------------------------------------------
  -- UUT
  -----------------------------------------------------------------------------

  component edge_detector is 
    generic (
      Nu : integer := Nu;
      Nh : integer := Nh;
      Ny : integer := Ny;
      n  : integer := n;
      m  : integer := m
      );
    port (
      clk  : in  STD_LOGIC;
      SE   : in  STD_LOGIC;
      WSI  : in  STD_LOGIC;
      b    : in  STD_LOGIC_VECTOR(m+n+1 downto 1);
      u    : in  STD_LOGIC_VECTOR(Nu*(m+n+1) downto 1);
      edge : out STD_LOGIC
      );
  end component;
  
  -------------------------------------------------------------------------------
  -- Private Constants
  -------------------------------------------------------------------------------

  constant bm    : std_logic_vector(m-1 downto 0) := (others => '0');
  constant bn    : std_logic_vector(n-1 downto 0) := (others => '0');
  constant num_w : integer := ((Nu+1)*Nh+((Nh+1)*Ny));
  
  -----------------------------------------------------------------------------
  -- Signals
  -----------------------------------------------------------------------------

  -- Inputs
  signal clk   :  STD_LOGIC;
  signal SE    :  STD_LOGIC;
  signal WSI   :  STD_LOGIC;
  signal b     :  STD_LOGIC_VECTOR(m+n+1 downto 1) := bm & "1" & bn;
  signal u     :  STD_LOGIC_VECTOR(Nu*(m+n+1) downto 1);

  -- Outputs 
  signal edge  : STD_LOGIC;
  type one_d_memory is array(1 to img_width) of
    std_logic_vector(m+n+1 downto 1);
  type two_d_memory is array(1 to img_height) of one_d_memory;
  signal u_all : STD_LOGIC_VECTOR(Nu*(m+n+1) downto 1);
  
  signal debugSignal: std_logic := '0';

-------------------------------------------------------------------------------
-- Begin the testbench
-------------------------------------------------------------------------------

begin
  
  -----------------------------------------------------------------------------
  -- Instantite the UUT
  -----------------------------------------------------------------------------
  

  uut: edge_detector
    generic map (
      Nu => Nu,
      Nh => Nh,
      Ny => Ny,
      m  => m,
      n  => n
      )
    port map (
      clk  => clk,
      SE   => SE,
      WSI  => WSI,
      b    => b,
      u    => u,
      edge => edge
      );
  
  -----------------------------------------------------------------------------
  -- Process for the clock
  -----------------------------------------------------------------------------
  
  clk_process: process
   begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
   end process;
  
  -----------------------------------------------------------------------------
  -- Process for the simulation
  -----------------------------------------------------------------------------
  
  sim_proc: process
    
    ---------------------------------------------------------------------------
    -- Process initializations
    ---------------------------------------------------------------------------
    
    -- Files
    file weight_file      : text open read_mode  is weight_path;
    file image_in         : text open read_mode  is in_path;
    file image_out        : text open write_mode is out_path;
    
    -- Variables
    variable out_line     : line;
    variable image_line   : line;
    variable weight_line  : line;
    variable value,value2 : STD_LOGIC_VECTOR(m+n+1 downto 1); 
    variable edge_out     : integer;
    variable end_of_line  : boolean;
    variable image_mem    : two_d_memory := (others => (others =>
      (others => '0')));
    variable weight       : STD_LOGIC_VECTOR(((Nu+1)*Nh+((Nh+1)*Ny))*(m+n+1)
      downto 1);
    variable i            : integer:=1;
    variable j            : integer:=1;
    variable k            : integer:=0;

begin
	
    wait for clk_period/2;
    ---------------------------------------------------------------------------
    -- Read in the weights
    ---------------------------------------------------------------------------

    while not endfile(weight_file) loop
      readline(weight_file, weight_line);    -- First line  
      read(weight_line, value, end_of_line); -- First value of the current line
      while end_of_line loop
        weight((num_w -k)*(m+n+1) downto (num_w -k-1)*(m+n+1)+1) := value;
        k := k + 1;
        read(image_line,value,end_of_line);
      end loop;
    end loop;
    
    ---------------------------------------------------------------------------
    -- Read image into matrix (image_mem)
    ---------------------------------------------------------------------------

    -- Reset counters
    i := 1;
    j := 1;
    
    -- Loop through image
    while not endfile(image_in) loop
      for j in 1 to img_width loop
        readline(image_in, image_line);
        read(image_line, value2, end_of_line);
        image_mem(i)(j)(m+n+1 downto 1) := value2;
      end loop;
      i := i + 1; 
      if (i > img_height) then
        i := 1;
      end if;
    end loop;
		
    ---------------------------------------------------------------------------
    -- Send weights to edge_detector
    ---------------------------------------------------------------------------
    
    SE <= '1';
    for k in (num_w*(m+n+1)) downto 1 loop
      WSI <= weight(k);
      wait for clk_period;
    end loop;
    SE <= '0';
    
    ---------------------------------------------------------------------------
    -- Send windows and recieve the edges
    ---------------------------------------------------------------------------
    
    i := 1; 
    while (i <= img_height) loop
      j := 1; 
      while (j <= img_width) loop
        u_all((Nu-j+1)*(m+n+1) downto (Nu-j)*(m+n+1) + 1) <=
					image_mem(i)(j)(m+n+1 downto 1);
        j := j + 1;
      end loop;
			
      wait for 8*clk_period;
      u <= u_all;
      wait for 7*clk_period;
			
      -- Capture edge of current window
      edge_out := conv_integer(edge);
			write(out_line, edge_out);      -- Write out edge
      writeline(image_out, out_line); -- Write out new row
      i := i + 1;
    end loop;
    wait;
   end process;
end;
