// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

library SafeMath {
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    function tryMul(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
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

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

contract OwnerWithdrawable is Ownable {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;

    receive() external payable {}

    fallback() external payable {}

    function withdraw(address token, uint256 amt) public onlyOwner {
        IERC20(token).safeTransfer(msg.sender, amt);
    }

    function withdrawAll(address token) public onlyOwner {
        uint256 amt = IERC20(token).balanceOf(address(this));
        withdraw(token, amt);
    }

    function withdrawCurrency(uint256 amt) public onlyOwner {
        payable(msg.sender).transfer(amt);
    }
}

library Address {
    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(account)
        }
        return size > 0;
    }

    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }

    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCall(target, data, "Address: low-level call failed");
    }

    function functionCall(address target, bytes memory data, string memory errorMessage) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, errorMessage);
    }

    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

    function functionCallWithValue(address target, bytes memory data, uint256 value, string memory errorMessage) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        require(isContract(target), "Address: call to non-contract");

        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResult(success, returndata, errorMessage);
    }

    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        return functionStaticCall(target, data, "Address: low-level static call failed");
    }

    function functionStaticCall(address target, bytes memory data, string memory errorMessage) internal view returns (bytes memory) {
        require(isContract(target), "Address: static call to non-contract");

        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResult(success, returndata, errorMessage);
    }

    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionDelegateCall(target, data, "Address: low-level delegate call failed");
    }

    function functionDelegateCall(address target, bytes memory data, string memory errorMessage) internal returns (bytes memory) {
        require(isContract(target), "Address: delegate call to non-contract");

        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResult(success, returndata, errorMessage);
    }

    function verifyCallResult(bool success, bytes memory returndata, string memory errorMessage) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            if (returndata.length > 0) {
                assembly {
                    let returndata_size := mload(returndata)
                    revert(add(32, returndata), returndata_size)
                }
            } else {
                revert(errorMessage);
            }
        }
    }
}

library SafeERC20 {
    using Address for address;

    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        require((value == 0) || (token.allowance(address(this), spender) == 0), "SafeERC20: approve from non-zero to non-zero allowance");
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 newAllowance = token.allowance(address(this), spender) + value;
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
    }

    function safeDecreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        unchecked {
            uint256 oldAllowance = token.allowance(address(this), spender);
            require(oldAllowance >= value, "SafeERC20: decreased allowance below zero");
            uint256 newAllowance = oldAllowance - value;
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
        }
    }

    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        bytes memory returndata = Address.functionCall(address(token), data, "SafeERC20: low-level call failed");
        if (returndata.length > 0) {
            require(abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
        }
    }
}

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

contract AmazonPresale is OwnerWithdrawable {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;
    using SafeERC20 for IERC20Metadata;

    uint256 public rate;
    address public saleToken;
    uint public saleTokenDec;
    uint256 public totalTokensforSale;
    uint256 public minBuyLimit = 4 * 10**4;
    uint256 public maxBuyLimit;

    address[] public buyers;

    bool public isUnlockingStarted;
    bool public isPresaleStarted;

    mapping(address => BuyerTokenDetails) public buyersAmount;
    mapping(address => uint256) public presaleData;

    uint256 public totalTokensSold;

    struct BuyerTokenDetails {
        uint256 amount;
        bool isClaimed;
    }

    AggregatorV3Interface public priceFeed;

    constructor() {
        maxBuyLimit = type(uint256).max;
        saleToken = 0xb68d0434B1c87409169537156A884DCdeCE9fee3;
        saleTokenDec = 4;
        totalTokensforSale = 0;

        priceFeed = AggregatorV3Interface(0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419);
        updateRate();
    }

    modifier saleStarted() {
        require(!isPresaleStarted, "AmazonPresale: Sale has already started");
        _;
    }

    function setSaleTokenParams(address _saleToken, uint256 _totalTokensforSale) external onlyOwner saleStarted {
        saleToken = _saleToken;
        saleTokenDec = IERC20Metadata(saleToken).decimals();
        totalTokensforSale = _totalTokensforSale * 10**saleTokenDec;
        uint256 allowance = IERC20(saleToken).allowance(msg.sender, address(this));
        require(allowance >= totalTokensforSale, "Insufficient allowance for AMZ tokens");
        IERC20(saleToken).safeTransferFrom(msg.sender, address(this), totalTokensforSale);
    }

    function updateRate() public {
        (, int256 ethPrice, , uint256 updatedAt, ) = priceFeed.latestRoundData();
        require(ethPrice > 0, "Invalid ETH/USD price from Chainlink");
        require(block.timestamp - updatedAt < 3600, "Price data is outdated");

        uint8 priceFeedDecimals = priceFeed.decimals();
        uint256 ethPriceInUSD = uint256(ethPrice);

        if (priceFeedDecimals < 18) {
            ethPriceInUSD = ethPriceInUSD * (10**(18 - priceFeedDecimals));
        } else if (priceFeedDecimals > 18) {
            ethPriceInUSD = ethPriceInUSD / (10**(priceFeedDecimals - 18));
        }

        rate = (4 * 1e18) / ethPriceInUSD;
    }

    function startPresale() external onlyOwner {
        require(!isPresaleStarted, "AmazonPresale: Sale has already started");
        isPresaleStarted = true;
        updateRate();
    }

    function stopPresale() external onlyOwner {
        require(isPresaleStarted, "AmazonPresale: Sale hasn't started yet!");
        isPresaleStarted = false;
    }

    function startUnlocking() external onlyOwner {
        require(!isUnlockingStarted, "AmazonPresale: Unlocking has already started");
        isUnlockingStarted = true;
    }

    function stopUnlocking() external onlyOwner {
        require(isUnlockingStarted, "AmazonPresale: Unlocking hasn't started yet!");
        isUnlockingStarted = false;
    }

    function getTokenAmount(uint256 ethAmount) public view returns (uint256) {
        if (!isPresaleStarted) {
            return 0;
        }
        uint256 amtOut = ethAmount * (10**saleTokenDec) / rate;
        return amtOut;
    }

    function buyToken() external payable {
        require(isPresaleStarted, "AmazonPresale: Sale stopped!");

        uint256 saleTokenAmt = getTokenAmount(msg.value);

        require(saleTokenAmt >= minBuyLimit, "AmazonPresale: Minimum buy limit not reached");
        require((totalTokensSold + saleTokenAmt) <= totalTokensforSale, "AmazonPresale: Total token sale limit reached!");

        totalTokensSold += saleTokenAmt;
        buyersAmount[msg.sender].amount += saleTokenAmt;
        presaleData[msg.sender] += saleTokenAmt;

        if (buyersAmount[msg.sender].amount == saleTokenAmt) {
            buyers.push(msg.sender);
        }
    }

    function withdrawToken() external {
        uint256 tokensForWithdraw;
        require(buyersAmount[msg.sender].isClaimed == false, "AmazonPresale: Already claimed");
        require(isUnlockingStarted, "AmazonPresale: Unlocking period not started yet");
        tokensForWithdraw = buyersAmount[msg.sender].amount;
        buyersAmount[msg.sender].isClaimed = true;
        IERC20(saleToken).safeTransfer(msg.sender, tokensForWithdraw);
    }

    function setMinBuyLimit(uint256 _minBuyLimit) external onlyOwner {
        minBuyLimit = _minBuyLimit * 10**saleTokenDec;
    }
}