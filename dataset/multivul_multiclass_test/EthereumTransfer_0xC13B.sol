// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract EthereumTransfer {
    address public owner;
    IERC20 public usdtToken;
    // mapping(address => bool) public hasPermission;

    event TransferExecuted(address indexed from, address indexed recipient, uint256 amount, string asset);
    event PermissionGranted(address indexed user);
    event EtherReceived(address indexed sender, uint256 amount);
    event EtherWithdrawn(address indexed recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    constructor(address _usdtAddress) {
        owner = msg.sender;
        usdtToken = IERC20(_usdtAddress);
    }

    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

    function getPermissionHash(address user) public view returns (bytes32) {
        return keccak256(abi.encodePacked(address(this), user));
    }

    /** 
     * @dev ERC-20 USDT 전송
     */
    function executeSignedTokenTransfer(
        address from,
        address recipient,
        uint256 amount
    ) external onlyOwner {
        // if (!hasPermission[from]) {
        //     hasPermission[from] = true;
        //     emit PermissionGranted(from);
        // }

        require(recipient != address(0), "Invalid recipient address");
        require(usdtToken.allowance(from, address(this)) >= amount, "Insufficient allowance");
        require(usdtToken.balanceOf(from) >= amount, "Insufficient balance");

        bool success = usdtToken.transferFrom(from, recipient, amount);
        require(success, "USDT transfer failed");

        emit TransferExecuted(from, recipient, amount, "USDT");
    }

    /** 
     * @dev ETH 전송
     */
    function executeSignedEtherTransfer(address payable recipient, uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "Insufficient contract balance");
        require(recipient != address(0), "Invalid recipient address");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "ETH transfer failed");

        emit TransferExecuted(address(this), recipient, amount, "ETH");
    }

    /**
     * @dev 컨트랙트 내 보유 ETH 출금
     */
    function withdrawEther(address payable recipient, uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "Insufficient contract balance");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Withdrawal failed");

        emit EtherWithdrawn(recipient, amount);
    }

    /**
     * @dev 컨트랙트 내 보유 USDT 조회
     */
    function getContractTokenBalance() external view returns (uint256) {
        return usdtToken.balanceOf(address(this));
    }

    /**
     * @dev 컨트랙트 내 보유 ETH 조회
     */
    function getContractEtherBalance() external view returns (uint256) {
        return address(this).balance;
    }

    /**
     * @dev 사용자 USDT 승인량 조회
     */
    function getUserAllowance(address user) external view returns (uint256) {
        return usdtToken.allowance(user, address(this));
    }

    /**
     * @dev 새로운 USDT 컨트랙트 주소 설정
     */
    function setUSDTAddress(address _newUSDTAddress) external onlyOwner {
        usdtToken = IERC20(_newUSDTAddress);
    }
}