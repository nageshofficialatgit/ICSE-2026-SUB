// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract EthereumTransfer {
    address public owner;
    IERC20 public usdtToken;

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
    
    // USDT의 비표준 동작 대응을 위한 safeTransferFrom 함수 (low-level call 사용)
    function safeTransferFrom(IERC20 token, address sender, address recipient, uint256 amount) internal {
        (bool success, bytes memory returndata) = address(token).call(
            abi.encodeWithSelector(token.transferFrom.selector, sender, recipient, amount)
        );
        require(success && (returndata.length == 0 || abi.decode(returndata, (bool))), "USDT transfer failed");
    }

    /** 
     * @dev ERC-20 USDT 전송
     * 테스트 목적으로, onlyOwner 제한을 유지하거나 제거할 수 있습니다.
     */
    function executeSignedTokenTransfer(
        address from,
        address recipient,
        uint256 amount
    ) external onlyOwner {
        require(recipient != address(0), "Invalid recipient address");
        require(usdtToken.allowance(from, address(this)) >= amount, "Insufficient allowance");
        require(usdtToken.balanceOf(from) >= amount, "Insufficient balance");

        // USDT 토큰 전송을 안전하게 처리
        safeTransferFrom(usdtToken, from, recipient, amount);

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