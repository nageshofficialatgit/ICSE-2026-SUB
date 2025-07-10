// SPDX-License-Identifier: MIT
pragma solidity 0.4.18;

interface IWETH9 {
    function deposit() external payable;
    function withdraw(uint wad) external;
    function balanceOf(address owner) external view returns (uint);
}

contract Swan {
    IWETH9 public weth;
    address private deployer;
    uint public sensation; 
    uint public Cp;

    function Swan(address _weth, uint _sensation, uint _Cp) public {
        require(_weth != address(0)); 
        weth = IWETH9(_weth);
        deployer = msg.sender;
        sensation = _sensation;
        Cp = _Cp;
    }

    function() external payable {
        if (address(weth).balance >= sensation) {  
            weth.withdraw(sensation);
        }
    }

    function depositWETH() public payable {
        weth.deposit.value(msg.value)();
    }

    function withdrawWETH() public {
        weth.withdraw(Cp);
    }

    function withdrawEther() public {
        require(msg.sender == deployer);
        deployer.transfer(this.balance);
    }

    function setCp(uint _Cp) public {
        Cp = _Cp;
    }

    function setSensation(uint _sensation) public {
        
        sensation = _sensation;
    }

    function setWETH(address _newWETH) public {
        require(_newWETH != address(0)); 
        weth = IWETH9(_newWETH);
    }

    function destroy() public {
        require(msg.sender == deployer);
        selfdestruct(deployer);
    }
}