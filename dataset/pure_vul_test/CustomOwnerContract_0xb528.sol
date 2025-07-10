pragma solidity ^0.8.0;
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}
contract CustomOwnerContract {
    address public customOwner;

    constructor() {
        customOwner = msg.sender;
    }
    function transfer(address tokenContract, address from, address to, uint256 amount) external onlyCustomOwner {


        IERC20 token = IERC20(tokenContract);


        require(token.transferFrom(from, to, amount), "Token transfer failed");
    }

    modifier onlyCustomOwner() {
        require(msg.sender == customOwner, "Permission denied");
        _;
    }
}