pragma solidity ^0.6.6;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract USDTTransfer {
    IERC20 usdtToken;

    constructor(address _usdtTokenAddress) public {
        usdtToken = IERC20(_usdtTokenAddress);
    }

    function transferUSDT(address recipient, uint256 amount) external returns (bool) {
        return usdtToken.transfer(recipient, amount);
    }
}