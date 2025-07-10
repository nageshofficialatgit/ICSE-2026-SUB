// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// --- Ownable.sol implementation ---
contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// --- LumeFI contract ---
contract LumeFI is Ownable {
    uint256 public constant INITIAL_SUPPLY = 1_000_000 * 10**18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    address public liquidityPool;

    string public name = "LumeFI";
    string public symbol = "LFI";
    uint8 public decimals = 18;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event LiquidityPoolUpdated(address newPool);
    event TransactionFeeUpdated(uint256 newFee);

    // Skickar ägaradressen till Ownable-konstruktorn
    constructor(address owner) Ownable() {
        transferOwnership(owner);  // Skickar rätt ägaradress till Ownable
        balanceOf[owner] = INITIAL_SUPPLY;
        totalSupply = INITIAL_SUPPLY;
    }

    // Funktion för att sätta en likviditetspool (endast ägaren kan göra detta)
    function setLiquidityPool(address pool) external onlyOwner {
        liquidityPool = pool;
        emit LiquidityPoolUpdated(pool);
    }

    // Hämtar transaktionsavgiften beroende på användarens saldo
    function getTransactionFee(address user) public view returns (uint256) {
        uint256 balance = balanceOf[user];
        if (balance >= 100_000 * 10**18) {
            return 5; // 0.5%
        } else if (balance >= 50_000 * 10**18) {
            return 10; // 1%
        } else if (balance >= 10_000 * 10**18) {
            return 20; // 2%
        } else {
            return 30; // 3%
        }
    }

    // Överföring av token
    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    // Godkänn en spenderare att överföra tokens från din adress
    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Utför en överföring från en adress (med godkännande)
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        uint256 currentAllowance = allowance[sender][msg.sender];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        allowance[sender][msg.sender] = currentAllowance - amount;
        _transfer(sender, recipient, amount);
        return true;
    }

    // Implementera vår egen version av _transfer
    function _transfer(address sender, address recipient, uint256 amount) internal {
        uint256 feePercentage = getTransactionFee(sender);
        uint256 fee = (amount * feePercentage) / 1000;
        uint256 amountAfterFee = amount - fee;

        // Skicka avgiften till likviditetspoolen
        if (fee > 0 && liquidityPool != address(0)) {
            balanceOf[liquidityPool] += fee;
            emit Transfer(sender, liquidityPool, fee);
        }
        balanceOf[sender] -= amount;
        balanceOf[recipient] += amountAfterFee;
        emit Transfer(sender, recipient, amountAfterFee);
    }
}