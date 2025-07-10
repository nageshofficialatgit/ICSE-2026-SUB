// SPDX-License-Identifier: MIT

pragma solidity ^0.8.26;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

interface IERC1363 is IERC20, IERC165 {
    function transferAndCall(address to, uint256 value) external returns (bool);
    function transferAndCall(address to, uint256 value, bytes calldata data) external returns (bool);
    function transferFromAndCall(address from, address to, uint256 value) external returns (bool);
    function transferFromAndCall(address from, address to, uint256 value, bytes calldata data) external returns (bool);
    function approveAndCall(address spender, uint256 value) external returns (bool);
    function approveAndCall(address spender, uint256 value, bytes calldata data) external returns (bool);
}

library SafeERC20 {
    error SafeERC20FailedOperation(address token);
    error SafeERC20FailedDecreaseAllowance(address spender, uint256 currentAllowance, uint256 requestedDecrease);

    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transfer, (to, value)));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transferFrom, (from, to, value)));
    }

    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeCall(token.approve, (spender, value));
        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeCall(token.approve, (spender, 0)));
            _callOptionalReturn(token, approvalCall);
        }
    }

    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            let success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            if iszero(success) {
                let ptr := mload(0x40)
                returndatacopy(ptr, 0, returndatasize())
                revert(ptr, returndatasize())
            }
            returnSize := returndatasize()
            returnValue := mload(0)
        }

        if (returnSize == 0 ? address(token).code.length == 0 : returnValue != 1) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        bool success;
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            returnSize := returndatasize()
            returnValue := mload(0)
        }
        return success && (returnSize == 0 ? address(token).code.length > 0 : returnValue == 1);
    }
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;
    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) revert OwnableInvalidOwner(address(0));
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) revert OwnableUnauthorizedAccount(_msgSender());
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) revert OwnableInvalidOwner(address(0));
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// Contract: LFCKICO

contract LFCKICO is Ownable {
    using SafeERC20 for IERC20;

    IERC20 public lfckToken;
    IERC20 public usdcToken;

    uint256 public tokenPriceUSD = 10; // 10 cents
    uint256 public constant DECIMALS = 1e18;
    uint256 public tokensSold;
    uint256 public maxTokensForSale = 35_000_000 * DECIMALS;
    uint256 public minPurchase = 1000 * DECIMALS;
    uint256 public maxPurchase = 500_000 * DECIMALS;
    bool public saleActive = true;

    mapping(address => uint256) public purchases;

    event TokensPurchased(address indexed buyer, uint256 amount, string currency);
    event SaleClosed();

    constructor(address _lfckToken, address _usdcToken) Ownable(msg.sender) {
        lfckToken = IERC20(_lfckToken);
        usdcToken = IERC20(_usdcToken);
    }

    receive() external payable {
        buyWithETH();
    }

    function buyWithETH() public payable {
        require(saleActive, "ICO is not active");
        require(msg.value > 0, "No ETH sent");

        uint256 ethPriceUSD = getETHPrice(); // Mock price
        uint256 usdValue = (msg.value * ethPriceUSD) / 1e18;
        uint256 tokensToBuy = (usdValue * DECIMALS) / tokenPriceUSD / 1e2;

        _processPurchase(tokensToBuy);
        emit TokensPurchased(msg.sender, tokensToBuy, "ETH");
    }

    function buyWithUSDC(uint256 usdcAmount) external {
        require(saleActive, "ICO is not active");
        require(usdcAmount > 0, "No USDC sent");

        uint256 tokensToBuy = (usdcAmount * DECIMALS) / tokenPriceUSD / 1e2;

        usdcToken.safeTransferFrom(msg.sender, address(this), usdcAmount);
        _processPurchase(tokensToBuy);
        emit TokensPurchased(msg.sender, tokensToBuy, "USDC");
    }

    function _processPurchase(uint256 tokensToBuy) internal {
        require(tokensSold + tokensToBuy <= maxTokensForSale, "Exceeds total allocation");
        require(purchases[msg.sender] + tokensToBuy >= minPurchase, "Below minimum purchase");
        require(purchases[msg.sender] + tokensToBuy <= maxPurchase, "Exceeds max purchase");

        purchases[msg.sender] += tokensToBuy;
        tokensSold += tokensToBuy;

        lfckToken.safeTransfer(msg.sender, tokensToBuy);
    }

    function getETHPrice() public pure returns (uint256) {
        return 3000 * 1e18; // Mock ETH price
    }

    function setPrice(uint256 newPriceInCents) external onlyOwner {
        tokenPriceUSD = newPriceInCents;
    }

    function pauseSale() external onlyOwner {
        saleActive = false;
        emit SaleClosed();
    }

    function resumeSale() external onlyOwner {
        saleActive = true;
    }

    function withdrawETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }

    function withdrawUSDC() external onlyOwner {
        uint256 balance = usdcToken.balanceOf(address(this));
        usdcToken.safeTransfer(owner(), balance);
    }

    function withdrawUnsoldTokens() external onlyOwner {
        uint256 remaining = lfckToken.balanceOf(address(this));
        lfckToken.safeTransfer(owner(), remaining);
    }
}