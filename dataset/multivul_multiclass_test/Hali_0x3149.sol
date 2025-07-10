// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract Hali {
    // 代币基本信息
    string public name = "Hali Token"; // 代币名称
    string public symbol = "HALI"; // 代币符号
    uint8 public decimals = 2; // 代币精度（小数点后2位）
    uint256 public rate = 1000; // 1 ETH = 1,000 Hali
    uint256 public totalSupply; // 代币总供应量
    address public owner; // 记录合约的所有者

    // 账户余额映射
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance; // 记录各地址允许他人转账的代币数量

    mapping(address => uint256) public airdropAmounts; // 存储每个用户的空投数量
    uint256 public totalAirdropTokens; // 记录合约中的空投总量

    // 定义事件
    event Transfer(address indexed from, address indexed to, uint256 value); // ERC-20 Transfer 事件
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    ); // ERC-20 Approval 事件
    event BoughtTokens(address indexed buyer, uint256 amount); //预售
    event WithdrawETH(address indexed to, uint256 amount); //转出合约中的eth

    event AirdropAssigned(address indexed recipient, uint256 amount); // 记录分配情况
    event AirdropClaimed(address indexed recipient, uint256 amount); // 记录领取情况

    event Stake(address indexed staker, uint256 amount); //质押
    event Unstake(address indexed staker, uint256 principal, uint256 interest); //赎回

    // 仅限合约所有者的修饰符
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    // 构造函数：初始化代币总量并分配给合约部署者
    constructor(uint256 totalAmount) payable {
        owner = msg.sender; // 记录合约的部署者
        balanceOf[owner] = totalAmount; // 赋值初始余额
        totalSupply = totalAmount; // 记录总供应量

        // 预留 90% 代币给合约，用于后续的代币销售
        uint256 reserveTokens = (totalAmount * 90) / 100;
        balanceOf[owner] -= reserveTokens;
        balanceOf[address(this)] = reserveTokens;
    }

    // ERC-20: 直接从调用者账户向 recipient 地址发送 amount 个代币
    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance"); // 确保余额充足
        require(to != address(0), "Invalid recipient"); // 确保目标地址有效

        balanceOf[msg.sender] -= amount; // 扣除发送者余额
        balanceOf[to] += amount; // 增加接收者余额

        emit Transfer(msg.sender, to, amount); // 触发转账事件
        return true;
    }

    // ERC-20: 允许 spender（被授权地址） 地址最多花费 amount 个代币
    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount; // 设置允许spender最大可花费金额
        emit Approval(msg.sender, spender, amount); // 触发批准事件
        return true;
    }

    // ERC-20: 从 sender（授权人地址） 地址向 recipient（接收者地址） 地址发送 amount 个代币
    function transferFrom(
        address _sender,
        address recipient,
        uint256 amount
    ) public returns (bool) {
        require(balanceOf[_sender] >= amount, "Insufficient balance"); // 确保 sender 账户余额充足
        require(
            allowance[_sender][address(this)] >= amount,
            "Allowance exceeded"
        ); // 确保 spender 有足够的授权

        balanceOf[_sender] -= amount; // 扣除 sender 余额
        balanceOf[recipient] += amount; // 增加 recipient 余额
        allowance[_sender][address(this)] -= amount; // 扣除 spender 的授权

        emit Transfer(_sender, recipient, amount); // 触发转账事件
        return true;
    }

    // 允许用户通过支付 ETH 购买 Hali 代币
    function buyTokens() public payable returns (bool) {
        require(msg.value >= 0.001 ether, "Must send at least 0.001 ETH");

        uint256 tokensToBuy = (msg.value * rate * (10**decimals)) / (10**18);
        require(tokensToBuy > 0, "ETH amount too low for any tokens");
        require(
            balanceOf[address(this)] >= tokensToBuy,
            "Not enough tokens in contract"
        );

        balanceOf[address(this)] -= tokensToBuy;
        balanceOf[msg.sender] += tokensToBuy;

        emit BoughtTokens(msg.sender, tokensToBuy);
        return true;
    }

    // 允许用户直接转账 ETH 购买代币
    receive() external payable {
        buyTokens();
    }

    // 允许合约所有者提取合约中的 ETH 资金
    function withdrawETH(address payable _to, uint256 _amount)
        public
        onlyOwner
    {
        require(
            _amount <= address(this).balance,
            "Insufficient balance in contract"
        ); // 确保合约内有足够资金

        (bool success, ) = _to.call{value: _amount}(""); // 使用 call 发送 ETH
        require(success, "Failed to send ETH"); // 确保交易成功

        emit WithdrawETH(_to, _amount); // 触发提现事件
    }

    // 允许用户销毁一定数量的代币（减少总供应量）
    function burn(uint256 amount) public {
        require(
            balanceOf[msg.sender] >= amount,
            "Insufficient balance to burn"
        );
        balanceOf[msg.sender] -= amount; // 扣除持有者的余额
        totalSupply -= amount; // 减少总供应量
    }

    // 允许合约的所有者铸造新的代币
    function mint(address recipient, uint256 amount) public onlyOwner {
        balanceOf[recipient] += amount; // 增加 recipient 的余额
        totalSupply += amount; // 增加总供应量
    }

    // -------- 质押功能 --------
    struct StakeInfo {
        uint256 amount; // 用户质押的 HALI 币数量
        uint256 startTime; // 质押的开始时间（时间戳）
    }

    // 记录每个用户的质押信息
    mapping(address => StakeInfo) public stakes;
    uint256 public dailyInterestRate = 13700; // 0.137% (单位 10^4, 实际为 13700 / 10^4)

    /**
     * @dev 由合约所有者（owner）修改每日利率
     * @param newRate 新的日化利率（单位 10**6，计算时除以 10**8）
     */
    function setDailyInterestRate(uint256 newRate) public onlyOwner {
        dailyInterestRate = newRate;
    }

    /**
     * @dev 用户进行质押操作
     * @param amount 质押的 HALI 币数量
     */
    function stake(uint256 amount) public {
        require(
            balanceOf[msg.sender] >= amount,
            "Insufficient balance to stake"
        ); // 检查用户余额是否足够
        require(
            stakes[msg.sender].amount == 0,
            "You must unstake first before staking again"
        ); // 确保没有进行中的质押

        balanceOf[msg.sender] -= amount; // 扣除用户的 HALI 余额
        stakes[msg.sender] = StakeInfo(amount, block.timestamp); // 记录质押金额和开始时间

        emit Stake(msg.sender, amount); // 触发质押事件
    }

    /**
     * @dev 用户赎回质押及利息
     */
    function unstake() public {
        StakeInfo storage userStake = stakes[msg.sender]; // 获取用户的质押信息
        require(userStake.amount > 0, "No active stake"); // 确保用户有质押

        uint256 stakingDays = (block.timestamp - userStake.startTime) / 1 days; // 计算质押的天数
        uint256 interest = (userStake.amount *
            dailyInterestRate *
            stakingDays) / (10**4); // 计算总利息

        // 避免极小值
        if (interest < 1) {
            interest = 0;
        }

        uint256 totalRedeem = userStake.amount + interest; // 计算总的可赎回数量（本金 + 利息）

        // 清除用户的质押记录
        stakes[msg.sender] = StakeInfo(0, 0);

        airdropAmounts[msg.sender] += totalRedeem; // 记录空投（利息 + 本金）
        totalAirdropTokens += totalRedeem; // 更新空投总量

        emit Unstake(msg.sender, userStake.amount, interest); // 触发赎回事件
        emit AirdropAssigned(msg.sender, totalRedeem); // 触发空投分配事件
    }

    // 空投代币
    // 空投代币1 Owner 分配空投（**支持累加，不会覆盖**）
    function assignAirdrop(address[] memory accounts, uint256[] memory amounts)
        public
        onlyOwner
    {
        require(accounts.length == amounts.length, "Mismatched arrays");

        for (uint256 i = 0; i < accounts.length; i++) {
            airdropAmounts[accounts[i]] += amounts[i]; // **累加空投数量**
            totalAirdropTokens += amounts[i]; // 记录总空投量
            emit AirdropAssigned(accounts[i], amounts[i]); // 触发事件
        }
    }

    // 空投代币2 用户自己领取空投，自己支付 gas
    function claimAirdrop() public {
        uint256 amount = airdropAmounts[msg.sender]; // 获取分配的空投数量
        require(amount > 0, "No airdrop available"); // 确保有空投
        require(
            balanceOf[address(this)] >= amount,
            "Not enough tokens in contract"
        );

        balanceOf[msg.sender] += amount; // 领取空投
        balanceOf[address(this)] -= amount;
        airdropAmounts[msg.sender] = 0; // 清零，防止重复领取
        totalAirdropTokens -= amount; // 更新合约中的空投总量

        emit AirdropClaimed(msg.sender, amount); // 触发事件
    }
}